package models.algorithm

import models.core.LinearModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}

/**
  * Created by WeiChen on 2016/5/26.
  */
class LogisticRegression extends LinearModel {
  override def hyperParameterTuning(data: RDD[LabeledPoint],test:RDD[LabeledPoint], iteration: List[Int] = List(10, 100, 1000), threshold: List[Double]): List[((Double, Double), LogisticRegressionModel, (Int, Double))] = {
    println("Total threshold:" + threshold.length)
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data)
    val hyper = threshold.map { t =>
      println("\n現在的threshold: " + t)
      val acc = accurate(model.clearThreshold().setThreshold(t), test)
      (acc,model, (0,t))
    }
    hyper
  }
}