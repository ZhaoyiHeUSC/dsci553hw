import scala.io.Source
import scala.util.Random
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

object BFRClustering {
  
  case class ClusterStats(N: Int, SUM: DenseVector[Double], SUMSQ: DenseVector[Double])
  
  def loadData(inputFile: String): Array[Array[Double]] = {
    Source.fromFile(inputFile).getLines().map { line =>
      line.split(",").map(_.toDouble)
    }.toArray
  }

  def saveOutput(outputFile: String, intermediateResults: Seq[String], clusteringResults: Seq[(Int, Int)]): Unit = {
    val writer = new java.io.PrintWriter(outputFile)
    writer.write("The intermediate results\n")
    intermediateResults.foreach(writer.println)
    writer.write("\n")
    writer.write("The clustering results\n")
    clusteringResults.foreach {
      case (idx, cluster) => writer.println(s"$idx,$cluster")
    }
    writer.close()
  }

  def mahalanobisDistance(point: DenseVector[Double], centroid: DenseVector[Double], covMatrix: DenseMatrix[Double]): Double = {
    try {
      val covInv = inv(covMatrix)
      val diff = point - centroid
      sqrt(diff.t * covInv * diff)
    } catch {
      case _: Exception =>
        val regCovMatrix = covMatrix + 1e-6 * DenseMatrix.eye[Double](covMatrix.rows)
        val covInv = inv(regCovMatrix)
        val diff = point - centroid
        sqrt(diff.t * covInv * diff)
    }
  }

  def updateStatistics(clusterStats: ClusterStats, points: Array[DenseVector[Double]]): ClusterStats = {
    val newN = clusterStats.N + points.length
    val newSum = clusterStats.SUM + points.map(_.toArray).reduce((a, b) => a.zip(b).map { case (x, y) => x + y }).toDenseVector
    val newSumSq = clusterStats.SUMSQ + points.map(p => p :* p).reduce((a, b) => a + b)
    ClusterStats(newN, newSum, newSumSq)
  }

  def runKMeans(data: Array[DenseVector[Double]], nClusters: Int): KMeansModel = {
    val spark = SparkSession.builder().appName("BFRClustering").getOrCreate()
    val df = spark.createDataFrame(data.map(x => (Vectors.dense(x.toArray), 0))).toDF("features", "label")
    val kmeans = new KMeans().setK(nClusters).setSeed(42)
    val model = kmeans.fit(df)
    model
  }

  def initializeBFR(data: Array[DenseVector[Double]], nClusters: Int): (KMeansModel, Array[DenseVector[Double]]) = {
    val nSamples = data.length
    val sampleSize = (0.2 * nSamples).toInt
    val sampledData = Random.shuffle(data.toList).take(sampleSize).toArray

    val kmeans = runKMeans(sampledData, nClusters * 5)

    val rsPoints = sampledData.zip(kmeans.transform(sampledData).collect()).filter {
      case (_, label) => kmeans.transform(Array(label)).collect().length == 1
    }.map(_._1)

    val kmeansFinal = runKMeans(sampledData, nClusters)
    (kmeansFinal, rsPoints)
  }

  def processBatch(data: Array[DenseVector[Double]], nClusters: Int, DSClusters: Map[Int, ClusterStats], CSClusters: Map[Int, ClusterStats], RSPoints: Array[DenseVector[Double]], kmeansLabels: Array[Int]): (Int, Int, Int, Int) = {
    val (discardPoints, compressionPoints, retainedPoints) = data.zip(kmeansLabels).foldLeft((Seq.empty[DenseVector[Double]], Seq.empty[DenseVector[Double]], Seq.empty[DenseVector[Double]])) {
      case ((discard, compression, retained), (point, label)) =>
        if (DSClusters.contains(label)) {
          (discard :+ point, compression, retained)
        } else if (CSClusters.contains(label)) {
          (discard, compression :+ point, retained)
        } else {
          (discard, compression, retained :+ point)
        }
    }

    (discardPoints.length, CSClusters.size, compressionPoints.length, retainedPoints.length)
  }

  def mergeClusters(DSClusters: Map[Int, ClusterStats], CSClusters: Map[Int, ClusterStats], dThreshold: Double): Seq[(ClusterStats, ClusterStats)] = {
    CSClusters.flatMap {
      case (csClusterId, csStats) =>
        DSClusters.collect {
          case (dsClusterId, dsStats) =>
            val dist = mahalanobisDistance(csStats.SUM, dsStats.SUM, DenseMatrix.eye[Double](dsStats.SUM.length)) // Simplified for demonstration
            if (dist < dThreshold) Some((csStats, dsStats))
            else None
        }.flatten
    }.toSeq
  }

  def iterativeBFR(data: Array[DenseVector[Double]], nClusters: Int, DSClusters: Map[Int, ClusterStats], CSClusters: Map[Int, ClusterStats], RSPoints: Array[DenseVector[Double]]): (Map[Int, ClusterStats], Map[Int, ClusterStats], Array[DenseVector[Double]]) = {
    val batchData = Random.shuffle(data.toList).take((0.2 * data.length).toInt).toArray
    val updatedDSClusters = DSClusters
    val updatedCSClusters = CSClusters
    val updatedRSPoints = RSPoints

    // Mahalanobis distance assignment logic for DS and CS would go here
    
    (updatedDSClusters, updatedCSClusters, updatedRSPoints)
  }

  def bfrClustering(inputFile: String, nClusters: Int, outputFile: String): Unit = {
    val data = loadData(inputFile)
    val totalDataPoints = data.length
    val dimensions = data.head.length - 2
    val chunkSize = totalDataPoints / 5
    val shuffledData = Random.shuffle(data.toList).toArray

    var DS = Map.empty[Int, ClusterStats]
    var CS = Map.empty[Int, ClusterStats]
    var RS = Array.empty[DenseVector[Double]]
    
    var intermediateResults = Seq.empty[String]
    var clusteringResults = Seq.empty[(Int, Int)]
    
    for (roundIdx <- 0 until 5) {
      val startIdx = roundIdx * chunkSize
      val endIdx = if (roundIdx < 4) (roundIdx + 1) * chunkSize else totalDataPoints
      val chunk = shuffledData.slice(startIdx, endIdx).map(d => DenseVector(d.tail))

      if (roundIdx == 0) {
        // Initial KMeans with large K
        val kmeans = runKMeans(chunk, nClusters * 5)
        val labels = kmeans.predict(chunk.map(_.toArray)).collect()
        
        labels.zip(chunk).foreach {
          case (label, point) =>
            if (labels.count(_ == label) == 1) {
              RS = RS :+ point
            } else {
              DS = DS.updatedWith(label) {
                case Some(stats) => Some(updateStatistics(stats, Array(point)))
                case None => Some(ClusterStats(1, point, point :* point))
              }
            }
        }
      } else {
        // Handle new points and assign to DS using Mahalanobis distance
      }

      // Merge CS clusters with Mahalanobis distance
      val mergedCS = mergeClusters(DS, CS, 2.0)
      
      // Save intermediate results
      intermediateResults = intermediateResults :+ s"Round $roundIdx: ${DS.values.map(_.N).sum}, ${CS.size}, ${CS.values.map(_.N).sum}, ${RS.length}"
    }

    // Final clustering result
    clusteringResults = DS.flatMap {
      case (clusterId, stats) =>
        val centroid = stats.SUM / stats.N.toDouble
        data.collect {
          case (id, point) if mahalanobisDistance(DenseVector(point.tail), centroid, DenseMatrix.eye[Double](stats.SUM.length)) < 2.0 =>
            (id.toInt, clusterId)
        }
    }.toSeq

    saveOutput(outputFile, intermediateResults, clusteringResults)
  }

  def main(args: Array[String]): Unit = {
    val inputFile = args(0)
    val nClusters = args(1).toInt
    val outputFile = args(2)
    
    bfrClustering(inputFile, nClusters, outputFile)
  }
}
