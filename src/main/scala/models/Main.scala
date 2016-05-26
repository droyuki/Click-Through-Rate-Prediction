package models

import models.algorithm.LogisticRegression
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import util.DataReader

/**
  * Created by WeiChen on 2016/5/26.
  */
object Main {
  def main(args: Array[String]) {
    val targetFeatures = List(
      "banner_pos", "site_id",  "site_category",
      "app_domain", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"
    )
    val data = new DataReader("/Users/WeiChen/Downloads/mid.csv")
      .readData()
      .selectFeatures(targetFeatures)
      .getLabelPoint()
      .randomSplit(Array(0.9, 0.1))
    LogManager.getRootLogger.setLevel(Level.ERROR)
    val trainData = data(0)
    val testData = data(1)

    println("開始訓練模型.....")
    val lr = new LogisticRegression
    val models = lr.hyperParameterTuning(trainData,testData,List(0),List(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0))
    val bestPar = lr.findBestModel(List(models))
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainData)
    val result = model.clearThreshold().setThreshold(bestPar._3._2)
    println("--------------------------------")
    println("Best Threshold: " + bestPar._3._2)
    lr.accurate(result,testData)

  }
}
