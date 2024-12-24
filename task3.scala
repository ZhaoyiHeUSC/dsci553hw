import scala.util.Random
import java.io.{BufferedWriter, FileWriter}
import scala.collection.mutable.ListBuffer

// Reservoir Sampling Function
def reservoirSampling(stream: Seq[String], reservoirSize: Int = 100): Unit = {
  var reservoir = ListBuffer[String]()  // Initialize an empty reservoir
  var globalSequence = 0  // Keep track of the sequence number
  val writer = new BufferedWriter(new FileWriter("reservoir_sampling_results.csv"))

  // Process each user in the stream
  for (user <- stream) {
    globalSequence += 1

    if (globalSequence <= reservoirSize) {
      // Fill the reservoir with the first 100 users
      reservoir += user
    } else {
      // Decide whether to replace a user in the reservoir
      val probability = reservoirSize.toDouble / globalSequence
      if (Random.nextDouble() < probability) {
        // Randomly pick an index to replace
        val replaceIdx = Random.nextInt(reservoirSize)
        reservoir(replaceIdx) = user
      }
    }

    // After processing each batch of 100 users, save the reservoir state
    if (globalSequence % reservoirSize == 0) {
      writer.write(s"$globalSequence,${reservoir(0)},${reservoir(20)},${reservoir(40)},${reservoir(60)},${reservoir(80)}\n")
    }
  }

  writer.close()
  println("Reservoir sampling results saved to reservoir_sampling_results.csv")
}

// Main function
def main(): Unit = {
  Random.setSeed(553)  // Seed the random generator
  val streamSize = 100
  val numOfAsks = 30
  val numUsers = 1000  // Simulated number of unique users

  for (_ <- 1 to numOfAsks) {
    // Generate a stream of 100 users
    val stream = (1 to streamSize).map(_ => s"user${Random.nextInt(numUsers) + 1}")
    reservoirSampling(stream)
  }
}

// Execute main function
main()
