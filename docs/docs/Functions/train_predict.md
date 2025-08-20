# TrainPredict Class 


The `train_predict.py` provides a full pipeline for training, evaluating, and saving machine learning models for churn prediction. It handles data loading, preprocessing, feature engineering, dataset balancing, model training, evaluation, and saving.

The default model used is a `RandomForestClassifier`, and the evaluation includes confusion matrix, ROC curve, precision-recall curve, accuracy, and a classification report.

---

## Methods

#### - `evaluate_model(model_name, model, X_test, y_test)`

Evaluates a trained model and generates visualizations and reports.

**Parameters:**

| Parameter    | Type                | Description                                            |
| ------------ | ------------------- | ------------------------------------------------------ |
| `model_name` | `str`               | Name of the model to be displayed in plots and report. |
| `model`      | `sklearn estimator` | Trained model object.                                  |
| `X_test`     | `pd.DataFrame`      | Features of the test set.                              |
| `y_test`     | `pd.Series`         | True target values of the test set.                    |

**Outputs:**

* Confusion matrix heatmap.
* ROC curve and AUC score.
* Precision-Recall curve.
* Accuracy score and classification report saved as text.
* Evaluation figure saved as PNG in `model_dir (MODELS_DIR = PROJ_ROOT / "models")`:

  * `<model_name>_evaluation.png`
  * `<model_name>_evaluation.txt`

---

#### - `pipeline()`

Runs the complete training and evaluation pipeline.

**Steps:**

1. **Load Dataset:** Reads the input CSV file and prints the shape.
2. **Split Data:** Splits dataset into training and testing sets.
3. **Clean Training Data:** Applies `CleanDataset().clean()` on `X_train`.
4. **Feature Engineering:** Applies `FeatureEng().feature_eng()` on `X_train`.
5. **Target Transformation:** Maps `Churn` column to binary values (`No -> 0, Yes -> 1`).
6. **SMOTE Balancing:** Balances the training dataset to handle class imbalance.
7. **Clean Test Data:** Applies cleaning and feature engineering on `X_test`.
8. **Train Model:** Fits a `RandomForestClassifier` on balanced training data.
9. **Evaluate Model:** Calls `evaluate_model()` on test data.
10. **Save Model:** Saves the trained model as `rf_model.joblib` in `model_dir (MODELS_DIR = PROJ_ROOT / "models")`.

**Outputs:**

* Trained model file: `rf_model.joblib`
* Evaluation PNG: `RandomForestClassifier_evaluation.png`
* Evaluation report TXT: `RandomForestClassifier_evaluation.txt`

---


### Notes

* The class currently supports only a `RandomForestClassifier` for training, but it can be extended to other models.
* All plots and reports are saved automatically in the model directory.
* Other models tested such as XGBoost, DecisionTree and LGBMCLassifier can be found in `notebooks/02_model_final.ipynb`
---

