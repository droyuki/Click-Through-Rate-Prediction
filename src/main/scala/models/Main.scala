package models

import models.algorithm.{LogisticRegression, SVM}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import util.DataReader

/**
  * Created by WeiChen on 2016/5/26.
  */
object Main {
  def getValidationData(data: RDD[LabeledPoint]): List[(RDD[LabeledPoint], RDD[LabeledPoint])] = {
    val test = for {i <- 1 to 10} yield {
      val Array(train, test) = data.randomSplit(Array(0.9, 0.1))
      (train, test)
    }
    test.toList
  }

  def main(args: Array[String]) {
    val targetFeatures = List(
      "site_id", "banner_pos", "C1", "hour", "C21"
    )
    val Array(trainData, testData) = new DataReader("/Users/WeiChen/Downloads/mid.csv")
      .readData()
      .selectFeatures(targetFeatures)
      .getLabelPoint()
      .randomSplit(Array(0.8, 0.2))

    LogManager.getRootLogger.setLevel(Level.ERROR)
    println("開始訓練模型.....")

    val splitData = getValidationData(trainData)

    //LR
    //    val lr = new LogisticRegression
    //    val lrModels = lr.hyperParameterTuning(splitData,List(0),List(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
    //    val bestParameter = lr.findBestModel(lrModels)
    //
    //    val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainData)
    //    println("\nHyperparameter complete.\n----------------------")
    //    println("Best threshold: " + bestParameter._4._2)
    //    val lrResult = lrModel.clearThreshold().setThreshold(bestParameter._4._2)
    //    lr.accurate(lrResult,testData)

    //SVM
    val svm = new SVM
    val svmModels = svm.hyperParameterTuning(splitData, List(10, 20), List(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    val bestPar = svm.findBestModel(svmModels)
    val svmModel = SVMWithSGD.train(trainData, bestPar._4._1)
    svmModel.clearThreshold()

    val scoreAndLabels = trainData.map { point =>
      val score = svmModel.predict(point.features)
      (score, point.label)
    }

    val min = scoreAndLabels.min()(new Ordering[Tuple2[Double, Double]]() {
      override def compare(x: (Double, Double), y: (Double, Double)): Int =
        Ordering[Double].compare(x._1, y._1)
    })

    val max = scoreAndLabels.max()(new Ordering[Tuple2[Double, Double]]() {
      override def compare(x: (Double, Double), y: (Double, Double)): Int =
        Ordering[Double].compare(x._1, y._1)
    })

    val svmResult = svmModel.clearThreshold().setThreshold(min._1 + (max._1 - min._1) * bestPar._4._2)
    svm.accurate(svmResult, testData)
    println("Threshold: " + bestPar._4._2)
  }
}
