package com.atguigu.offline

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.jblas.DoubleMatrix

object OfflineRecommenderTest {
  val MONGO_UI = "mongodb://dk100:27017/recommender"
  val MONGODB_RATING_COLLECTION = "Rating"
  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"

  case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: Int )

  // 定义一个基准推荐对象
  case class Recommendation( mid: Int, score: Double )

  // 定义基于预测评分的用户推荐列表
  case class UserRecs( uid: Int, recs: Seq[Recommendation] )

  // 定义基于LFM电影特征向量的电影相似度列表
  case class MovieRecs( mid: Int, recs: Seq[Recommendation] )


  def main(args: Array[String]): Unit = {

    //1、spark读取mongodb数据 用户行为数据
    val spark = SparkSession.builder()
      .appName("OfflineRecommenderTest")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    val inputRDD = spark.read
      .format("com.mongodb.spark.sql")
      .option("uri", MONGO_UI)
      .option("collection", MONGODB_RATING_COLLECTION)
      .load()
      .as[MovieRating]
      .rdd
      .map(a => Rating(a.uid,a.mid,a.score))

    //2、用ALS算法，传入行为数据，训练出模型，并根据指定用户物品集合预测出结果
    val userRDD: RDD[Int] = inputRDD.map(rating => rating.user).distinct()
    val movieRDD: RDD[Int] = inputRDD.map(rating => rating.product).distinct()
    val ratingRDD: RDD[(Int, Int)] = userRDD.cartesian(movieRDD)

    val model: MatrixFactorizationModel = ALS.train(inputRDD, 200, 5, 0.1)
    val predictRDD: RDD[Rating] = model.predict(ratingRDD)

    val userRecsRDD: RDD[UserRecs] = predictRDD.map(predict => (predict.user, (predict.product, predict.rating)))
      .groupByKey()
      .map { case (userId, iter) => {
        val recommendations: List[Recommendation] =
//          iter.toList.sortWith((a, b) => a._2 >= b._2).take(20).map { case (mid, score) => Recommendation(mid, score) }
          iter.toList.sortWith(_._2 >= _._2).take(20).map { case (mid, score) => Recommendation(mid, score) }
        UserRecs(userId, recommendations)
      }}

    //3、将给用户的推荐列表数据存入mongoDB
    //    userRecsRDD.collect().foreach(println)
    userRecsRDD.toDF()
      .write
      .format("com.mongodb.spark.sql")
      .option("uri",MONGO_UI)
      .option("collection",USER_RECS)
      .mode("overwrite")
      .save()

    //4、根据模型得到物品的特征向量，求出物品的相似度(余弦)
    val productFeatures: RDD[(Int, Array[Double])] = model.productFeatures
    val cartesianRDD: RDD[((Int, Array[Double]), (Int, Array[Double]))] = productFeatures.cartesian(productFeatures)

    val movieSimRDD: RDD[MovieRecs] = cartesianRDD
      .filter { case (a, b) => a._1 != b._1 }
      .map {
        case ((midA,arrayA), (midB,arrayB)) => {
          val score: Double = getSimilarity(arrayA, arrayB)
          (midA, (midB, score))
        }
      }
      .filter(_._2._2 > 0.6)
      .groupByKey()
      .map {
        case (userId, iter) => MovieRecs(
          userId,
//          iter.toList.sortWith { case (a, b) => a._2 > b._2 }.map { case (mid, score) => Recommendation(mid, score) }
          iter.toList.sortWith { _._2 > _._2 }.map { case (mid, score) => Recommendation(mid, score) }
        )
      }

    movieSimRDD.toDF()
      .write
      .format("com.mongodb.spark.sql")
      .option("uri",MONGO_UI)
      .option("collection",MOVIE_RECS)
      .mode("overwrite")
      .save()

  }

  /**
   * 求两个特征向量的相似度
   * @param a
   * @param b
   * @return
   */
  def getSimilarity(a: Array[Double], b: Array[Double]) = {
    val matrixA = new DoubleMatrix(a)
    val matrixB = new DoubleMatrix(b)
    val score = matrixA.dot(matrixB)/(matrixA.norm2()*matrixB.norm2())
    score
  }

}
