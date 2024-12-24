import java.security.MessageDigest
import scala.util.Random
import java.io._

object FlajoletMartin {

  // Generate hash values for a given user_id
  def myhashs(userId: String): List[BigInt] = {
    var hashList: List[BigInt] = List()
    for (i <- 0 until 3) { 
      val md5 = MessageDigest.getInstance("MD5")
      val hashBytes = md5.digest(userId.getBytes("UTF-8"))
      val hashValue = BigInt(hashBytes) + i
      hashList = hashList :+ hashValue
    }
    hashList
  }

  // Flajolet-Martin sketch class
  class FlajoletMartin(numHashes: Int = 3) {
    private val maxZeros = Array.fill(numHashes)(0)

    // Add a user to the sketch
    def add(userId: String): Unit = {
      val hashValues = myhashs(userId)
      for (i <- 0 until numHashes) {
        val trailingZeros = countTrailingZeros(hashValues(i))
        maxZeros(i) = math.max(maxZeros(i), trailingZeros)
      }
    }

    // Estimate the cardinality
    def estimate(): Double = {
      val estimates = maxZeros.map(z => math.pow(2, z))
      estimates.sum / estimates.length
    }

    // Count trailing zeros in a number
    private def countTrailingZeros(n: BigInt): Int = {
      var count = 0
      var num = n
      while (num % 2 == 0 && num != 0) {
        count += 1
        num /= 2
      }
      count
    }
  }

  // Main execution function
  def main(args: Array[String]): Unit = {
    val streamSize = 300
    val numOfAsks = 30
    var results = List[(Int, Int, Double)]()

    for (batch <- 0 until numOfAsks) {
      val flajoletMartin = new FlajoletMartin()
      var userSet = Set[String]()

      // Simulate a stream of user IDs
      for (_ <- 0 until streamSize) {
        val userId = s"user_${Random.nextInt(1000) + 1}"
        flajoletMartin.add(userId)
        userSet += userId
      }

      val groundTruth = userSet.size
      val estimation = flajoletMartin.estimate()
      results = results :+ (batch, groundTruth, estimation)
    }

    // Save results to CSV
    val writer = new PrintWriter(new File("flajolet_martin_results.csv"))
    writer.write("Time,Ground Truth,Estimation\n")
    results.foreach { case (batch, groundTruth, estimation) =>
      writer.write(s"$batch,$groundTruth,$estimation\n")
    }
    writer.close()

    // Calculate and print the ratio for validation
    val totalGroundTruth = results.map(_._2).sum
    val totalEstimations = results.map(_._3).sum
    val ratio = if (totalGroundTruth > 0) totalEstimations / totalGroundTruth else 0

    if (ratio >= 0.2 && ratio <= 5) {
      println(s"Ratio is within acceptable limits: $ratio")
    } else {
      println(s"Ratio is out of acceptable limits: $ratio")
    }
  }
}
