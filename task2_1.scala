import org.apache.spark.{SparkConf, SparkContext}
import scala.math._
import java.io._

object task2_1 {
  def main(args: Array[String]): Unit = {
    val train_file_name = args(0)
    val test_file_name = args(1)
    val output_file_name = args(2)

    val conf = new SparkConf().setAppName("task2_1").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val train_lines = sc.textFile(train_file_name)
    val train_header = train_lines.first()
    val train_data = train_lines.filter(_ != train_header)
      .map(_.split(','))
      .map(x => (x(0), x(1), x(2).toDouble))

    val test_lines = sc.textFile(test_file_name)
    val test_header = test_lines.first()
    val test_data = test_lines.filter(_ != test_header)
      .map(_.split(','))
      .map(x => (x(0), x(1)))

    val user_business_ratings = train_data.map { case (user, business, rating) =>
      (user, (business, rating))
    }.groupByKey()
      .mapValues(_.toMap)
      .collectAsMap()

    val business_user_ratings = train_data.map { case (user, business, rating) =>
      (business, (user, rating))
    }.groupByKey()
      .mapValues(_.toMap)
      .collectAsMap()

    val global_average = train_data.map(_._3).mean()

    val business_avg_ratings = train_data.map { case (user, business, rating) =>
      (business, rating)
    }.groupByKey()
      .mapValues(ratings => ratings.sum / ratings.size)
      .collectAsMap()

    val user_avg_ratings = train_data.map { case (user, business, rating) =>
      (user, rating)
    }.groupByKey()
      .mapValues(ratings => ratings.sum / ratings.size)
      .collectAsMap()

    val bc_user_business_ratings = sc.broadcast(user_business_ratings)
    val bc_business_user_ratings = sc.broadcast(business_user_ratings)
    val bc_user_avg_ratings = sc.broadcast(user_avg_ratings)
    val bc_business_avg_ratings = sc.broadcast(business_avg_ratings)
    val bc_global_average = sc.broadcast(global_average)

    val predictions = test_data.map { case (user_id, business_id) =>
      val prediction = predict_rating(
        user_id, business_id,
        bc_user_business_ratings.value,
        bc_business_user_ratings.value,
        bc_user_avg_ratings.value,
        bc_business_avg_ratings.value,
        bc_global_average.value)
      (user_id, business_id, prediction)
    }

    val predictions_list = predictions.collect()

    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output_file_name)))
    writer.write("user_id,business_id,prediction\n")
    for ((user_id, business_id, prediction) <- predictions_list) {
      writer.write(s"$user_id,$business_id,$prediction\n")
    }
    writer.close()
  }

  def compute_similarity(business1: String, business2: String,
                         business_user_ratings: scala.collection.Map[String, Map[String, Double]],
                         user_avg_ratings: scala.collection.Map[String, Double]): Option[Double] = {
    val ratings1 = business_user_ratings.getOrElse(business1, Map.empty[String, Double])
    val ratings2 = business_user_ratings.getOrElse(business2, Map.empty[String, Double])
    val common_users = ratings1.keySet.intersect(ratings2.keySet)
    val n_common = common_users.size
    if (n_common == 0) return None

    var numerator = 0.0
    var denominator1 = 0.0
    var denominator2 = 0.0
    for (u <- common_users) {
      val r1 = ratings1(u) - user_avg_ratings(u)
      val r2 = ratings2(u) - user_avg_ratings(u)
      numerator += r1 * r2
      denominator1 += r1 * r1
      denominator2 += r2 * r2
    }
    if (denominator1 == 0 || denominator2 == 0) return None
    val raw_similarity = numerator / sqrt(denominator1 * denominator2)
    // Apply significance weighting
    val significance = math.min(n_common, 50).toDouble / 50.0 // Cap at 50 users
    val adjusted_similarity = raw_similarity * significance
    Some(adjusted_similarity)
  }

  def predict_rating(user_id: String, business_id: String,
                     user_business_ratings: scala.collection.Map[String, Map[String, Double]],
                     business_user_ratings: scala.collection.Map[String, Map[String, Double]],
                     user_avg_ratings: scala.collection.Map[String, Double],
                     business_avg_ratings: scala.collection.Map[String, Double],
                     global_average: Double,
                     N: Int = 100): Double = {
    val user_ratings = user_business_ratings.getOrElse(user_id, Map.empty[String, Double])
    if (user_ratings.isEmpty) {
      // Cold start user: return business average or global average
      return business_avg_ratings.getOrElse(business_id, global_average)
    }
    var similarities = scala.collection.mutable.ArrayBuffer[(Double, Double)]()
    for (b <- user_ratings.keys) {
      val simOpt = compute_similarity(business_id, b, business_user_ratings, user_avg_ratings)
      if (simOpt.isDefined) {
        val sim = simOpt.get
        val rating_diff = user_ratings(b) - user_avg_ratings(user_id)
        similarities += ((sim, rating_diff))
      }
    }
    if (similarities.isEmpty) {
      // No similar items found: return user average or business average
      return user_avg_ratings.getOrElse(user_id, business_avg_ratings.getOrElse(business_id, global_average))
    }
    // Keep top N similarities
    val sorted_similarities = similarities.sortBy { case (sim, _) => -math.abs(sim) }.take(N)
    val numerator = sorted_similarities.map { case (sim, rating_diff) => sim * rating_diff }.sum
    val denominator = sorted_similarities.map { case (sim, _) => math.abs(sim) }.sum
    if (denominator == 0) {
      // Should not happen, but just in case
      return user_avg_ratings.getOrElse(user_id, global_average)
    }
    val predicted_rating = user_avg_ratings(user_id) + numerator / denominator
    // Clip the rating between 1 and 5
    math.max(1.0, math.min(5.0, predicted_rating))
  }
}
