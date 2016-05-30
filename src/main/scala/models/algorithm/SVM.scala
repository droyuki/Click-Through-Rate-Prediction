package models.algorithm

import models.core.LinearModel
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.rdd.RDD

/**
 * Created by WanEnFu on 16/5/26.
 */
class SVM extends LinearModel {

  override def hyperParameterTuning(dataList: List[(RDD[LabeledPoint],RDD[LabeledPoint])], iteration: List[Int], threshold: List[Double]): List[(Double, Double, Double, (Int, Double))] = {
    val crossValidation = dataList.map { data =>
       val hyperList = iteration.map(e => {
        val model = SVMWithSGD.train(data._1, e)
        model.clearThreshold()

        val scoreAndLabels = data._1.map { point =>
          val score = model.predict(point.features)
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
        (min._1, max._1, model, e)
      })
      .map(e1 => {
        val hypers = threshold.map(e2 => {
          val resetModel = e1._3.setThreshold(e1._1 + (e1._2 - e1._1) * e2)
          (accurate(resetModel, data._2), (e1._4, e2))
        })
        hypers
      })
      hyperList.flatten
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