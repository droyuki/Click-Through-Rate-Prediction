package models.core

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.rdd.RDD

/**
  * Created by WeiChen on 2016/5/26.
  */

abstract class LinearModel() extends Serializable {
  //dataList : List[(train, test)]
  def hyperParameterTuning(dataList: List[(RDD[LabeledPoint],RDD[LabeledPoint])], iteration: List[Int] = List(10, 100, 1000), threshold: List[Double]): List[(Double, Double, Double, (Int, Double))]

  final def accurate(model: GeneralizedLinearModel, test: RDD[LabeledPoint]): (Double, Double, Double) = {
    // Compute raw scores on the test set
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Precision threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Recall threshold: $t, Recall: $r")
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

    //correctNum = TP+TN
    val correctNum = predictionAndLabels.filter(pair => pair._1 != pair._2).count()
    println("TP+TN = " + correctNum)
    (auROC, auPRC, correctNum.toDouble)
  }

  def findBestModel(modelList: List[(Double, Double, Double, (Int, Double))]): (Double, Double,Double, (Int, Double)) = {
    //find by TP+TN
    val best = modelList.maxBy(a => a._3)
    best
  }
}
