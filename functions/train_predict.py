
### Imports ###
import joblib
import warnings
import pandas as pd
from joblib import load
from pathlib import Path

from dataset import CleanDataset
from features import FeatureEng

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc, 
    precision_recall_curve
)

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

warnings.filterwarnings('ignore')



class TrainPredict():
    def __init__(
        self, 
        input_path: Path = RAW_DATA_DIR / "churn_raw_data.csv",
        processed_dir: Path = PROCESSED_DATA_DIR,  
        model_dir: Path = MODELS_DIR,
        test_size: float = 0.2, 
        random_state: int = 42 
    ):
        self.input_path = input_path
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        self.test_size = test_size
        self.random_state = random_state
        self.cleaner = CleanDataset  
        self.featurizer = FeatureEng 
  
    def evaluate_model(self, 
                       model_name, 
                       model, 
                       X_test, 
                       y_test):
        
        # Predict the labels and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] 

        print(f"Model Evaluation: {model_name}\n")

        # Prepare the figure for plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, 
                    annot=True, 
                    fmt="d", 
                    cmap="Blues", 
                    cbar=False, 
                    ax=axes[0])
        axes[0].set_title(f'Confusion Matrix')
        axes[0].set_xlabel("Prediction")
        axes[0].set_ylabel("Real Value")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, 
                    tpr, 
                    color='darkorange', 
                    label=f'AUC = {roc_auc:.2f}')
        axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        axes[2].plot(recall, precision, color='green')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curve')
        axes[2].grid()

        plt.suptitle(model_name)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
        # save the figure
        fig.savefig(self.model_dir / f"{model_name}_evaluation.png")

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
     
        # Join the accuracy and classification report in a txt file
        with open(self.model_dir / f"{model_name}_evaluation.txt", "w") as f:
            f.write(f"Accuracy: {acc:.2f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred)) #type: ignore


    def pipeline(self):

        #----- Load the dataset -----#
        data = pd.read_csv(self.input_path)      
        print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
        
        # Split the data into training and testing sets
        X = data.drop(columns=['Churn'])
        y = data['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )

        #----- Process the training data -----#
        X_train = self.cleaner().clean(X_train) 

        # Feature Engineering
        X_train = self.featurizer().feature_eng(X_train)

        # Transform the target variable
        y_train = y_train.map({'No': 0, 'Yes': 1})

        
        # Smote for balancing the dataset
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train) #type: ignore
        print(f"Training data balanced: {X_train_bal.shape[0]} rows, {X_train_bal.shape[1]} columns.")
        
        # #----- Process the test data -----#
        X_test = self.cleaner().clean(X_test)

        # Feature Engineering
        X_test = self.featurizer().feature_eng(X_test)

        # Transform the target variable
        y_test = y_test.map({'No': 0, 'Yes': 1})
        
        #----- Train and Evaluate Model -----#
        # ! The evaluation of all models is done in the same way, in notebooks/02_model_final, 
        # here i opted to use only one model.
        
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train_bal, 
                     y_train_bal)
        
        self.evaluate_model("RandomForestClassifier", 
                            rf_model, 
                            X_test, 
                            y_test)
        
        # Save the model
        rf_model_path = self.model_dir / "rf_model.joblib"

        joblib.dump(rf_model, rf_model_path)
        
        print(f"Model and predictions saved at {rf_model_path}")
        

if __name__ == "__main__":
    trainer = TrainPredict()
    trainer.pipeline()
    