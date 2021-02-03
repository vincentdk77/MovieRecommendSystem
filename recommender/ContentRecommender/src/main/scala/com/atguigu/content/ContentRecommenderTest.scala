package com.atguigu.content

import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.jblas.DoubleMatrix

object ContentRecommenderTest {

  val MONGO_UI = "mongodb://dk100:27017/recommender"
  val MONGODB_MOVIE_COLLECTION = "Movie"
  val CONTENT_MOVIE_RECS = "ContentMovieRecs"

  // 需要的数据源是电影内容信息
  case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                   shoot: String, language: String, genres: String, actors: String, directors: String)

  // 定义一个基准推荐对象
  case class Recommendation( mid: Int, score: Double )

  // 定义电影内容信息提取出的特征向量的电影相似度列表
  case class MovieRecs( mid: Int, recs: Seq[Recommendation] )

  def main(args: Array[String]): Unit = {

    //1、读mongo中的物品数据
    val spark = SparkSession.builder().appName("ContentRecommenderTest").master("local[*]").getOrCreate()

    import spark.implicits._
    val inputRDD = spark.read
      .format("com.mongodb.spark.sql")
      .option("uri", MONGO_UI)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .load()
      .as[Movie]
      .map(movie =>(movie.mid,movie.name,movie.genres.replace("|"," ")))
      .toDF("mid","name","genres")

    //2、根据“电影类型”字段，运用tf-idf算法得到特征向量RDD
    val tokenizer: Tokenizer = new Tokenizer().setInputCol("genres").setOutputCol("words")
    val tokenizerDF: DataFrame = tokenizer.transform(inputRDD)

    val hashingTF: HashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(50)
    val tfDF: DataFrame = hashingTF.transform(tokenizerDF)
//    tfDF.show(false)

    val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel: IDFModel = idf.fit(tfDF)
    val idfDF: DataFrame = idfModel.transform(tfDF)
//    idfDF.show(false)

    val productFeatures: RDD[(Int, Array[Double])] = idfDF.map(row => {
      val mid: Int = row.getAs[Int]("mid")
      val denseVector = row.getAs[SparseVector]("features").toArray
      (mid, denseVector)
    }).rdd.cache()
//    println(productFeatures.count())

    //3、特征向量RDD的笛卡尔积，再经过转换，得到任意两个电影的相似度
    val cartesianRDD: RDD[((Int, Array[Double]), (Int, Array[Double]))] = productFeatures.cartesian(productFeatures)

    val movieSimRDD/*: RDD[MovieRecs]*/ = cartesianRDD
      .filter { case (a, b) => a._1 != b._1 }
      .map {
        case ((midA,arrayA), (midB,arrayB)) => {
          val score: Double = getSimilarity(arrayA, arrayB)
          (midA, (midB, score))
        }
      }
      .filter(_._2._2 > 0.6)
//    println(movieSimRDD.count())
      .groupByKey()
      .map {
        case (userId, iter) => MovieRecs(
          userId,
          //          iter.toList.sortWith { case (a, b) => a._2 > b._2 }.map { case (mid, score) => Recommendation(mid, score) }
          iter.toList.sortWith { _._2 > _._2 }.map { case (mid, score) => Recommendation(mid, score) }
        )
      }
//    println(movieSimRDD.count())

    movieSimRDD.toDF()
      .write
      .format("com.mongodb.spark.sql")
      .option("uri",MONGO_UI)
      .option("collection",CONTENT_MOVIE_RECS)
      .mode("overwrite")
      .save()

  }

  def getSimilarity(a: Array[Double], b: Array[Double]) = {
    val matrixA = new DoubleMatrix(a)
    val matrixB = new DoubleMatrix(b)
    val score = matrixA.dot(matrixB)/(matrixA.norm2()*matrixB.norm2())
    score
  }
}
