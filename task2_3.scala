import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import scala.collection.mutable
import scala.util.parsing.json.JSON
import java.io.PrintWriter
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost}
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit

object task2_3 {
  def main(args: Array[String]): Unit = {
    val folder_path = args(0)
    val test_file_name = args(1)
    val output_file_name = args(2)

    val conf = new SparkConf().setAppName("task2_3").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    // Read training data
    val train_file = folder_path + "/yelp_train.csv"
    val train_lines = sc.textFile(train_file)
    val train_header = train_lines.first()
    val train_rdd = train_lines.filter(_ != train_header).map(_.split(',')).map { case Array(user_id, business_id, rating) =>
      (user_id, business_id, rating.toDouble)
    }

    // Read test data
    val test_lines = sc.textFile(test_file_name)
    val test_header = test_lines.first()
    val test_rdd = test_lines.filter(_ != test_header).map(_.split(',')).map { case Array(user_id, business_id, _*) =>
      (user_id, business_id)
    }

    // Collect user_ids and business_ids
    val train_user_ids = train_rdd.map(_._1).distinct()
    val train_business_ids = train_rdd.map(_._2).distinct()
    val test_user_ids = test_rdd.map(_._1).distinct()
    val test_business_ids = test_rdd.map(_._2).distinct()

    val all_user_ids = train_user_ids.union(test_user_ids).distinct()
    val all_business_ids = train_business_ids.union(test_business_ids).distinct()

    val user_ids_set = all_user_ids.collect().toSet
    val business_ids_set = all_business_ids.collect().toSet

    // Read user.json
    val user_file = folder_path + "/user.json"
    val user_lines = sc.textFile(user_file)
    val user_rdd = user_lines.map(line => {
      val json = JSON.parseFull(line)
      json match {
        case Some(map: Map[String, Any]) =>
          val user_id = map("user_id").asInstanceOf[String]
          if (user_ids_set.contains(user_id)) (user_id, map) else null
        case _ => null
      }
    }).filter(_ != null)

    val user_feature_dict = user_rdd.collectAsMap()

    // Read business.json
    val business_file = folder_path + "/business.json"
    val business_lines = sc.textFile(business_file)
    val business_rdd = business_lines.map(line => {
      val json = JSON.parseFull(line)
      json match {
        case Some(map: Map[String, Any]) =>
          val business_id = map("business_id").asInstanceOf[String]
          if (business_ids_set.contains(business_id)) (business_id, map) else null
        case _ => null
      }
    }).filter(_ != null)

    val business_feature_dict = business_rdd.collectAsMap()

    // Default features
    val defaultUserFeature = Map[String, Any]("average_stars" -> 3.75, "review_count" -> 0, "useful" -> 0, "fans" -> 0, "cool" -> 0, "funny" -> 0,
      "elite" -> "", "yelping_since" -> "2010-01-01")

    val defaultBusinessFeature = Map[String, Any]("stars" -> 3.75, "review_count" -> 0, "is_open" -> 1, "categories" -> "", "attributes" -> Map[String, Any]())

    // Extract top categories
    def extractCategories(businessMap: Map[String, Any]): Array[String] = {
      val categories = businessMap.getOrElse("categories", "").toString
      categories.split(",").map(_.trim).filter(_.nonEmpty)
    }

    val all_categories = business_rdd.flatMap { case (_, businessMap) =>
      extractCategories(businessMap)
    }.map(category => (category, 1)).reduceByKey(_ + _)

    val top_categories = all_categories.sortBy(-_._2).map(_._1).take(50)
    val top_categories_broadcast = sc.broadcast(top_categories.toSet)

    // Function to extract features
    def extractFeatures(user_id: String, business_id: String): Array[Double] = {
      val user_feature = user_feature_dict.getOrElse(user_id, defaultUserFeature)
      val business_feature = business_feature_dict.getOrElse(business_id, defaultBusinessFeature)

      val features = mutable.ArrayBuffer[Double]()
      // User features
      features += user_feature.get("average_stars").map(_.toString.toDouble).getOrElse(3.75)
      features += user_feature.get("review_count").map(_.toString.toDouble).getOrElse(0.0)
      features += user_feature.get("useful").map(_.toString.toDouble).getOrElse(0.0)
      features += user_feature.get("funny").map(_.toString.toDouble).getOrElse(0.0)
      features += user_feature.get("fans").map(_.toString.toDouble).getOrElse(0.0)
      features += user_feature.get("cool").map(_.toString.toDouble).getOrElse(0.0)
      // New User Features
      val elite = user_feature.getOrElse("elite", "").toString
      val elite_count = if (elite != "" && elite != "None") elite.split(",").length else 0
      features += elite_count.toDouble

      val yelping_since_str = user_feature.getOrElse("yelping_since", "2010-01-01").toString
      val yelping_years = try {
        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
        val yelping_since = LocalDate.parse(yelping_since_str, formatter)
        val now = LocalDate.now()
        val years = ChronoUnit.DAYS.between(yelping_since, now) / 365.0
        years
      } catch {
        case _: Exception => 0.0
      }
      features += yelping_years

      // Business features
      features += business_feature.get("stars").map(_.toString.toDouble).getOrElse(3.75)
      features += business_feature.get("review_count").map(_.toString.toDouble).getOrElse(0.0)
      val is_open = business_feature.get("is_open").map(_.toString.toDouble).getOrElse(1.0)
      features += is_open

      // Additional business features
      val categories = business_feature.getOrElse("categories", "").toString
      val categories_list = categories.split(",").map(_.trim).filter(_.nonEmpty)
      val categories_set = categories_list.toSet

      val top_categories_set = top_categories_broadcast.value
      val category_features = top_categories.map { category =>
        if (categories_set.contains(category)) 1.0 else 0.0
      }
      features ++= category_features

      // Parse attributes
      val attributes = business_feature.get("attributes") match {
        case Some(attrMap: Map[String, Any]) => attrMap
        case _ => Map[String, Any]()
      }

      // RestaurantsPriceRange2
      val price_range = attributes.get("RestaurantsPriceRange2").map(_.toString.toDouble).getOrElse(2.0)
      features += price_range

      // Alcohol
      val alcohol = attributes.get("Alcohol").map(_.toString.toLowerCase)
      val alcohol_value = alcohol match {
        case Some(a) if a.contains("none") => 0.0
        case Some(_) => 1.0
        case _ => 1.0
      }
      features += alcohol_value

      // WiFi
      val wifi = attributes.get("WiFi").map(_.toString.toLowerCase)
      val wifi_value = wifi match {
        case Some(w) if w.contains("no") => 0.0
        case Some(_) => 1.0
        case _ => 1.0
      }
      features += wifi_value

      // NoiseLevel
      val noise_level = attributes.get("NoiseLevel").map(_.toString.toLowerCase)
      val noise_level_value = noise_level match {
        case Some(n) if n.contains("quiet") => 1.0
        case Some(n) if n.contains("average") => 2.0
        case Some(n) if n.contains("loud") => 3.0
        case Some(n) if n.contains("very_loud") => 4.0
        case _ => 2.0
      }
      features += noise_level_value

      // GoodForKids
      val good_for_kids = attributes.get("GoodForKids").map(_.toString.toLowerCase)
      val good_for_kids_value = good_for_kids match {
        case Some(g) if g.contains("true") => 1.0
        case _ => 0.0
      }
      features += good_for_kids_value

      // OutdoorSeating
      val outdoor_seating = attributes.get("OutdoorSeating").map(_.toString.toLowerCase)
      val outdoor_seating_value = outdoor_seating match {
        case Some(o) if o.contains("true") => 1.0
        case _ => 0.0
      }
      features += outdoor_seating_value

      // Interaction features
      val user_avg_stars = user_feature.get("average_stars").map(_.toString.toDouble).getOrElse(3.75)
      val business_avg_stars = business_feature.get("stars").map(_.toString.toDouble).getOrElse(3.75)
      features += user_avg_stars * business_avg_stars

      features.toArray
    }

    // Prepare training data
    val train_features_rdd = train_rdd.map { case (user_id, business_id, rating) =>
      val features = extractFeatures(user_id, business_id)
      (user_id, business_id, rating, features)
    }

    val trainFeaturesAndLabels = train_features_rdd.map { case (_, _, rating, features) =>
      (features.map(_.toFloat), rating.toFloat)
    }.collect()

    val trainFeaturesArray = trainFeaturesAndLabels.map(_._1)
    val trainLabelsArray = trainFeaturesAndLabels.map(_._2)

    val numTrainSamples = trainFeaturesArray.length
    val numFeatures = trainFeaturesArray(0).length
    val trainDataFlat = trainFeaturesArray.flatten

    val trainData = new DMatrix(trainDataFlat, numTrainSamples, numFeatures, Float.NaN)
    trainData.setLabel(trainLabelsArray)

    // Set XGBoost parameters
    val paramMap = Map(
      "objective" -> "reg:linear",
      "eta" -> 0.04,
      "max_depth" -> 8,
      "subsample" -> 0.9,
      "colsample_bytree" -> 0.9,
      "lambda" -> 1.0,
      "alpha" -> 1.0,
      "gamma" -> 0.2
    )

    val numRound = 400

    // Train the model
    val booster = XGBoost.train(trainData, paramMap, numRound)

    // Prepare test data
    val test_features_rdd = test_rdd.map { case (user_id, business_id) =>
      val features = extractFeatures(user_id, business_id)
      (user_id, business_id, features.map(_.toFloat))
    }

    val testFeaturesData = test_features_rdd.map { case (_, _, features) => features }.collect()

    val numTestSamples = testFeaturesData.length
    val testDataFlat = testFeaturesData.flatten

    val testData = new DMatrix(testDataFlat, numTestSamples, numFeatures, Float.NaN)

    // Predict with model-based RS
    val model_predictions = booster.predict(testData).map(_.head)

    val testDataPairs = test_features_rdd.map { case (user_id, business_id, _) => (user_id, business_id) }.collect()

    // Implement item-based Collaborative Filtering (CF)
    // Build user-business ratings dictionary
    val user_business_rating_rdd = train_rdd.map { case (user_id, business_id, rating) =>
      (user_id, (business_id, rating))
    }.groupByKey().mapValues(_.toMap)

    val user_business_rating_dict = user_business_rating_rdd.collectAsMap()

    // Build business-user ratings dictionary
    val business_user_rating_rdd = train_rdd.map { case (user_id, business_id, rating) =>
      (business_id, (user_id, rating))
    }.groupByKey().mapValues(_.toMap)

    val business_user_rating_dict = business_user_rating_rdd.collectAsMap()

    // Precompute item-item similarities (optional, may increase runtime)
    // For simplicity, we'll compute similarities on the fly and cache them
    val similarity_dict = mutable.Map[(String, String), Option[Double]]()

    // Function to compute similarity
    def computeSimilarity(business_id1: String, business_id2: String): Option[Double] = {
      val users1 = business_user_rating_dict.getOrElse(business_id1, Map.empty[String, Double])
      val users2 = business_user_rating_dict.getOrElse(business_id2, Map.empty[String, Double])
      val common_users = users1.keySet.intersect(users2.keySet)
      if (common_users.size < 3) {
        None
      } else {
        val ratings1 = common_users.map(u => users1(u)).toArray
        val ratings2 = common_users.map(u => users2(u)).toArray
        val avg1 = ratings1.sum / ratings1.length
        val avg2 = ratings2.sum / ratings2.length
        val numerator = ratings1.zip(ratings2).map { case (r1, r2) => (r1 - avg1) * (r2 - avg2) }.sum
        val denominator = math.sqrt(ratings1.map(r => (r - avg1) * (r - avg1)).sum) * math.sqrt(ratings2.map(r => (r - avg2) * (r - avg2)).sum)
        if (denominator == 0) None else Some(numerator / denominator)
      }
    }

    // Function to predict rating using CF
    def predictRatingCF(user_id: String, business_id: String): (Option[Double], Int) = {
      if (!user_business_rating_dict.contains(user_id) || !business_user_rating_dict.contains(business_id)) {
        (None, 0)
      } else {
        val user_ratings = user_business_rating_dict(user_id)
        val similarities = user_ratings.keys.filter(_ != business_id).flatMap { other_business_id =>
          val key = if (business_id < other_business_id) (business_id, other_business_id) else (other_business_id, business_id)
          val sim = similarity_dict.getOrElseUpdate(key, computeSimilarity(business_id, other_business_id))
          sim.map(s => (s, user_ratings(other_business_id)))
        }.filter(_._1 > 0).toSeq

        if (similarities.isEmpty) {
          (None, 0)
        } else {
          val sorted_similarities = similarities.sortBy { case (sim, _) => -sim }
          val top_n = sorted_similarities.take(100)
          val numerator = top_n.map { case (sim, rating) => sim * rating }.sum
          val denominator = top_n.map { case (sim, _) => math.abs(sim) }.sum
          if (denominator == 0) {
            (None, 0)
          } else {
            val cf_prediction = numerator / denominator
            val num_similar_items = top_n.length
            (Some(cf_prediction), num_similar_items)
          }
        }
      }
    }

    // Function to compute dynamic alpha
    def dynamicAlpha(num_similar_items: Int): Double = {
      if (num_similar_items >= 50) {
        0.7
      } else if (num_similar_items >= 20) {
        0.6
      } else if (num_similar_items >= 5) {
        0.5
      } else {
        0.3
      }
    }

    // Predict ratings with CF and combine with model-based predictions
    val hybrid_predictions = (0 until numTestSamples).map { i =>
      val user_id = testDataPairs(i)._1
      val business_id = testDataPairs(i)._2
      val model_pred = model_predictions(i)
      val (cf_pred_option, num_similar_items) = predictRatingCF(user_id, business_id)
      val final_pred = cf_pred_option match {
        case Some(cf_pred) =>
          val alpha = dynamicAlpha(num_similar_items)
          val combined_pred = alpha * cf_pred + (1 - alpha) * model_pred
          math.max(1.0, math.min(5.0, combined_pred))
        case None =>
          math.max(1.0, math.min(5.0, model_pred))
      }
      (user_id, business_id, final_pred)
    }

    // Write output
    val writer = new PrintWriter(output_file_name)
    writer.write("user_id,business_id,prediction\n")
    hybrid_predictions.foreach { case (user_id, business_id, prediction) =>
      writer.write(s"$user_id,$business_id,$prediction\n")
    }
    writer.close()

    sc.stop()
  }
}
