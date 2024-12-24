import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable
import scala.util.parsing.json.JSON
import java.io.PrintWriter
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost}

object task2_2 {
  def main(args: Array[String]): Unit = {
    val folder_path = args(0)
    val test_file_name = args(1)
    val output_file_name = args(2)

    val conf = new SparkConf().setAppName("task2_2").setMaster("local[*]")
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
        val yelping_since = java.time.LocalDate.parse(yelping_since_str)
        val now = java.time.LocalDate.now()
        val years = java.time.temporal.ChronoUnit.YEARS.between(yelping_since, now)
        years.toDouble
      } catch {
        case _: Exception => 0.0
      }
      features += yelping_years

      // Business features
      features += business_feature.get("stars").map(_.toString.toDouble).getOrElse(3.75)
      features += business_feature.get("review_count").map(_.toString.toDouble).getOrElse(0.0)
      val is_open = business_feature.get("is_open").map(_.toString.toDouble).getOrElse(1.0)
      features += is_open

      // Additional features can be added here...

      features.toArray
    }

    // Prepare training data
    val train_features_rdd = train_rdd.map { case (user_id, business_id, rating) =>
      val features = extractFeatures(user_id, business_id)
      (user_id, business_id, rating, features)
    }

    val trainFeaturesAndLabels = train_features_rdd.map { case (user_id, business_id, rating, features) =>
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

    val testFeaturesData = test_features_rdd.map { case (user_id, business_id, features) => features }.collect()

    val numTestSamples = testFeaturesData.length
    val testDataFlat = testFeaturesData.flatten

    val testData = new DMatrix(testDataFlat, numTestSamples, numFeatures, Float.NaN)

    // Predict
    val predictions = booster.predict(testData)

    // Write output
    val testDataPairs = test_features_rdd.map { case (user_id, business_id, features) => (user_id, business_id) }.collect()

    val writer = new PrintWriter(output_file_name)
    writer.write("user_id,business_id,prediction\n")
    for (i <- 0 until numTestSamples) {
      val user_id = testDataPairs(i)._1
      val business_id = testDataPairs(i)._2
      val prediction = predictions(i)(0)
      writer.write(s"$user_id,$business_id,$prediction\n")
    }
    writer.close()

    sc.stop()
  }
}
