package models.core

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.rdd.RDD

/**
  * Created by WeiChen on 2016/5/26.
  */

abstract class LinearModel() extends Serializable {
  //  def train(data: RDD[LabeledPoint],hyperParameters:D): GeneralizedLinearModel
  def hyperParameterTuning(data: RDD[LabeledPoint], test: RDD[LabeledPoint], iteration: List[Int] = List(10, 100, 1000), threshold:List[Double]): List[((Double, Double), GeneralizedLinearModel, (Int, Double))]

  final def accurate(model: GeneralizedLinearModel, test: RDD[LabeledPoint] ): (Double, Double) = {
    // Compute raw scores on the test set
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Precision Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Recall Threshold: $t, Recall: $r")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)

    (auROC, auPRC)
  }

  def findBestModel(modelMatrix: List[List[((Double,Double),GeneralizedLinearModel, (Int, Double))]]):(Double, Double, (Int, Double))={
    val transMatrix = modelMatrix.transpose
    val length = transMatrix(0).size
    val sumList = transMatrix.map(l => {
      val length = l.size
      var sumAUC = 0.0
      var sumPRC = 0.0
      l.foreach(e => {
        sumAUC = sumAUC + e._1._1
        sumPRC = sumPRC + e._1._2
      })
      (sumAUC/length, sumPRC/length, l(0)._3)
    })
    val best = sumList.maxBy((a)=> a._1)
    best
  }
}
