package com.atguigu.content

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.jblas.DoubleMatrix

/**
  * Copyright (c) 2018-2028 尚硅谷 All Rights Reserved 
  *
  * Project: MovieRecommendSystem
  * Package: com.atguigu.content
  * Version: 1.0
  *
  * Created by wushengran on 2019/4/4 9:14
  */

// 需要的数据源是电影内容信息
case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                 shoot: String, language: String, genres: String, actors: String, directors: String)

case class MongoConfig(uri:String, db:String)

// 定义一个基准推荐对象
case class Recommendation( mid: Int, score: Double )

// 定义电影内容信息提取出的特征向量的电影相似度列表
case class MovieRecs( mid: Int, recs: Seq[Recommendation] )

object ContentRecommender {

  // 定义表名和常量
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val CONTENT_MOVIE_RECS = "ContentMovieRecs"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://foo-1:27017/recommender",
      "mongo.db" -> "recommender"
    )

    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("OfflineRecommender")

    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    // 加载数据，并作预处理
    val movieTagsDF = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .map(
        // 提取mid，name，genres三项作为原始内容特征，分词器默认按照空格做分词
        x => ( x.mid, x.name, x.genres.map(c=> if(c=='|') ' ' else c) )
      )
      .toDF("mid", "name", "genres")
      .cache()

    // 核心部分： 用TF-IDF从内容信息中提取电影特征向量

    // 创建一个分词器，默认按空格分词
    val tokenizer: Tokenizer = new Tokenizer().setInputCol("genres").setOutputCol("words")

    // 用分词器对原始数据做转换，生成新的一列words
    val wordsData: DataFrame = tokenizer.transform(movieTagsDF)//输出: [action,sci-fi]    [drama,horror,thriller]

    // 引入HashingTF工具，可以把一个词语序列转化成对应的词频
    val hashingTF: HashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(50)//设置维度为50
    val featurizedData: DataFrame = hashingTF.transform(wordsData)//输出： (50,[40,46],[1.0,1.0])        (50,[26,27,36],[1.0,1.0,1.0])

    // 引入IDF工具，可以得到idf模型
    val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    // 训练idf模型，得到每个词的逆文档频率
    val idfModel: IDFModel = idf.fit(featurizedData)
    // 用模型对原数据进行处理，得到文档中每个词的tf-idf，作为新的特征向量
    val rescaledData: DataFrame = idfModel.transform(featurizedData)//输出（只有自己有的特征的维度的向量）： (50,[40,46],[1.82..,2.405...])        (50,[26,27,36],[2.1947..,0.7605..,1.7079..])

    // TODO: DF与DS的打印方式，   truncate = false 打印的时候输出全部，而不是超出长度显示"..."
//    rescaledData.show(truncate = false)

    val movieFeatures = rescaledData.map(
      row => ( row.getAs[Int]("mid"), row.getAs[SparseVector]("features").toArray )//SparseVector:指的是稀疏向量   (Vector.toArray后变成稠密向量)
    )
      .rdd
      .map(
        x => ( x._1, new DoubleMatrix(x._2) )
      )

    // TODO: rdd的打印方式
    movieFeatures.collect().foreach(println)  //输出（50个维度的稠密的向量）：(1889,[0.0; 0.0; 0.0; 0.0; 2.8046147489591897; 0.0; 0.0; 0.0; 0.0; 0.0;......])


    //下面部分与offlineRecommender一致！！

    // 对所有电影两两计算它们的相似度，先做笛卡尔积
    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter{
        // 把自己跟自己的配对过滤掉
        case (a, b) => a._1 != b._1
      }
      .map{
        case (a, b) => {
          val simScore = this.consinSim(a._2, b._2)
          ( a._1, ( b._1, simScore ) )
        }
      }
      .filter(_._2._2 > 0.6)    // 过滤出相似度大于0.6的
      .groupByKey()
      .map{
        case (mid, items) => MovieRecs( mid, items.toList.sortWith(_._2 > _._2).map(x => Recommendation(x._1, x._2)) )
      }
      .toDF()
    movieRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", CONTENT_MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
  }

  // 求向量余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix):Double ={
    movie1.dot(movie2) / ( movie1.norm2() * movie2.norm2() )
  }
}
