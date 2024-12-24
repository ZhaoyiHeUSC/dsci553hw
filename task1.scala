import java.io.{File, PrintWriter}
import scala.util.Random
import scala.math._

class BloomFilter(val size: Int, val numHashes: Int) {
  private val bitArray = Array.fill(size)(0)
  private val hashParams = generateHashParams()

  def generateHashParams(): Seq[(Int, Int, Int)] = {
    val primes = Seq(101, 103, 107, 109, 113, 127)
    for (_ <- 0 until numHashes) yield {
      val a = Random.nextInt(100) + 1
      val b = Random.nextInt(100)
      val p = primes(Random.nextInt(primes.size))
      (a, b, p)
    }
  }

  def hash(item: Long, a: Int, b: Int, p: Int): Int = {
    ((a * item + b) % p) % size
  }

  def convertToInt(userId: String): Long = {
    BigInt(userId.getBytes("UTF-8")).toLong
  }

  def hashAndSet(item: String): Unit = {
    val convertedItem = convertToInt(item)
    hashParams.foreach { case (a, b, p) =>
      val hashVal = hash(convertedItem, a, b, p)
      bitArray(hashVal) = 1
    }
  }

  def add(userId: String): Unit = {
    hashAndSet(userId)
  }

  def check(userId: String): Boolean = {
    val item = convertToInt(userId)
    hashParams.forall { case (a, b, p) =>
      bitArray(hash(item, a, b, p)) == 1
    }
  }
}

object BloomFilterModule {
  def evaluateFalsePositiveRate(bloomFilter: BloomFilter, numChecks: Int = 1000): Double = {
    val testUserIds = (1001 to 2000).map(i => s"test_user_$i").take(numChecks)
    val falsePositives = testUserIds.count(userId => bloomFilter.check(userId))
    falsePositives.toDouble / numChecks
  }

  def simulateBloomFilter(bitArrayLength: Int, numHashFunctions: Int, streamSize: Int, numOfAsks: Int, outputFile: String = "bloom_filter_results.csv"): Unit = {
    val bloomFilter = new BloomFilter(bitArrayLength, numHashFunctions)
    var previousUserSet = Set[String]()
    var results = Seq[(Double, Double)]()

    for (batchIndex <- 0 until numOfAsks) {
      val stream = (1 to streamSize).map(_ => s"user_${Random.nextInt(1000) + 1}")
      val falsePositives = stream.count(userId => bloomFilter.check(userId) && !previousUserSet.contains(userId))

      // Add elements to Bloom Filter and previous user set
      stream.foreach(userId => {
        bloomFilter.add(userId)
        previousUserSet += userId
      })

      // Calculate False Positive Rate (FPR)
      val fpr = falsePositives.toDouble / streamSize
      val elapsedTime = System.nanoTime().toDouble / 1e9
      results :+= (elapsedTime, fpr)
    }

    // Save results to CSV
    val writer = new PrintWriter(new File(outputFile))
    writer.write("Time,FPR\n")
    results.foreach { case (time, fpr) =>
      writer.write(f"$time%.4f,$fpr%.4f\n")
    }
    writer.close()

    println(s"Simulation complete. Results saved to '$outputFile'.")
  }

  def main(args: Array[String]): Unit = {
    val bitArrayLength = 69997
    val numHashFunctions = 5
    val streamSize = 100
    val numOfAsks = 30

    // Start the simulation
    simulateBloomFilter(bitArrayLength, numHashFunctions, streamSize, numOfAsks)
  }
}
