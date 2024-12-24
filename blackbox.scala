import scala.util.Random
import scala.io.Source

class BlackBox {

  def ask(file: String, num: Int): Array[String] = {
    val lines = Source.fromFile(file).getLines().toArray
    val users = Array.fill(num)("")

    for (i <- 0 until num) {
      users(i) = lines(Random.nextInt(lines.length)).stripLineEnd
    }

    users
  }
}

object Main extends App {
  val bx = new BlackBox()
  // val users = bx.ask("users.txt", 5)
  // println(users.mkString(", "))
}
