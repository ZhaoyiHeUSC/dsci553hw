import org.apache.spark.{SparkConf, SparkContext}
import scala.util.hashing.MurmurHash3
import scala.collection.mutable
import java.io._

object task1 {
  def main(args: Array[String]): Unit = {
    val input_file_name = args(0)
    val output_file_name = args(1)

    val conf = new SparkConf().setAppName("task1").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val lines = sc.textFile(input_file_name)
    val header = lines.first()
    val data = lines.filter(_ != header)

    val rdd = data.map(_.split(',')).map(x => (x(1), x(0))) // (business_id, user_id)

    val user_ids = data.map(_.split(',')(0)).distinct().collect()
    val num_users = user_ids.length
    val user_index = user_ids.zipWithIndex.toMap

    val business_user_rdd = data.map(_.split(',')).map(x => (x(1), user_index(x(0)))).groupByKey().mapValues(_.toSet)

    val business_user_dict = business_user_rdd.collectAsMap()
    val business_user_dict_b = sc.broadcast(business_user_dict)

    val n = 60
    val b = 30
    val r = 2

    val hash_funcs = generate_hash_funcs(n, num_users)

    val business_signatures = business_user_rdd.mapValues(user_indices => compute_minhash_signature(user_indices, hash_funcs, num_users))

    val candidate_pairs = business_signatures.flatMap { case (business_id, signatures) =>
      get_bands(business_id, signatures, b, r)
    }.groupByKey().map(_._2.toList).filter(_.size > 1).flatMap { list =>
      list.combinations(2).map { case List(a, b) => if (a < b) (a, b) else (b, a) }
    }.distinct()

    val results = candidate_pairs.map { case (b1, b2) =>
      val similarity = compute_jaccard(b1, b2, business_user_dict_b.value)
      (b1, b2, similarity)
    }.filter(_._3 >= 0.5)

    val sorted_results = results.map { case (b1, b2, sim) =>
      ((b1, b2), sim)
    }.sortByKey()

    val sorted_results_list = sorted_results.collect()

    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output_file_name)))
    writer.write("business_id_1,business_id_2,similarity\n")
    for (((b1, b2), sim) <- sorted_results_list) {
      writer.write(s"$b1,$b2,$sim\n")
    }
    writer.close()
  }

  def generate_hash_funcs(num_hashes: Int, max_user_id: Int): Array[(Int, Int, Int)] = {
    val random = new scala.util.Random(42)
    val max_int = Int.MaxValue
    val a_list = Array.fill(num_hashes)(random.nextInt(max_int - 1) + 1)
    val b_list = Array.fill(num_hashes)(random.nextInt(max_int))
    val p = 2333333333  // A large prime number
    a_list.zip(b_list).map { case (a, b) => (a, b, p) }
  }

  def compute_minhash_signature(user_indices: Iterable[Int], hash_funcs: Array[(Int, Int, Int)], num_users: Int): Array[Int] = {
    hash_funcs.map { case (a, b, p) =>
      user_indices.map { idx =>
        ((a.toLong * idx + b) % p % num_users).toInt
      }.min
    }
  }

  def get_bands(business_id: String, signatures: Array[Int], b: Int, r: Int): Seq[((Int, Seq[Int]), String)] = {
    (0 until b).map { i =>
      val start = i * r
      val end = start + r
      val band_signature = signatures.slice(start, end).toSeq
      ((i, band_signature), business_id)
    }
  }

  def compute_jaccard(b1: String, b2: String, business_user_dict: scala.collection.Map[String, Set[Int]]): Double = {
    val users1 = business_user_dict(b1)
    val users2 = business_user_dict(b2)
    val intersection = users1.intersect(users2).size
    val union = users1.union(users2).size
    intersection.toDouble / union
  }
}
