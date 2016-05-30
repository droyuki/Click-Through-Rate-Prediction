package models.algorithm

import models.core.LinearModel
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by WeiChen on 2016/5/26.
  */
class LogisticRegression extends LinearModel {
  //dataList : List[(train, test)]
  override def hyperParameterTuning(dataList:List[(RDD[LabeledPoint],RDD[LabeledPoint])], iteration: List[Int] = List(10, 100, 1000), threshold: List[Double]): List[(Double, Double, Double, (Int, Double))] = {
    val crossValidation = dataList.map { data =>
      val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data._1)
      val hyperList = threshold.map { t =>
        println("\n----------------------")
        (accurate(model.clearThreshold().setThreshold(t), data._2), (0, t))
      }
      hyperList
    }.transpose

    val sumList = crossValidation.map(l => {
      val length = l.size
      var sumAUC = 0.0
      var sumPRC = 0.0
      var sumCorrectNum = 0.0
      l.foreach(e => {
        sumAUC = sumAUC + e._1._1
        sumPRC = sumPRC + e._1._2
        sumCorrectNum = sumCorrectNum + e._1._3
      })
      (sumAUC/length, sumPRC/length, sumCorrectNum, l(0)._2)
    })
    sumList
  }
}