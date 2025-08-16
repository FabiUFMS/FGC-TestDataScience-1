### Imports ###
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class FeatureEng():
    def __init__(self, 
                 label_columns: list = ["gender", 
                                        "seniorcitizen", 
                                        "partner", 
                                        "dependents", 
                                        "phoneservice", 
                                        "paperlessbilling"], 
                 one_hot_columns: list = ["multiplelines", 
                                          "internetservice", 
                                          "onlinesecurity", 
                                          "onlinebackup", 
                                          "deviceprotection", 
                                          "techsupport", 
                                          "streamingtv", 
                                          "streamingmovies", 
                                          "paymentmethod"],
                 label_encoder: LabelEncoder = LabelEncoder(),
                 one_hot_encoder: OneHotEncoder = OneHotEncoder(sparse_output=False)):
        
        self.label_columns = label_columns
        self.one_hot_columns = one_hot_columns
        self.label_encoder = label_encoder
        self.one_hot_encoder = one_hot_encoder

    def feature_eng(self, 
                    data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering on the input DataFrame by applying label encoding, one-hot encoding, 
        and mapping categorical values to numerical representations.
        Steps performed:
            - Maps the 'contract' column values ('monthly', 'one year', 'two year') to integers (0, 1, 2).
            - Applies label encoding to columns specified in self.label_columns, if present in the DataFrame.
            - Applies one-hot encoding to columns specified in self.one_hot_columns.
            - Concatenates the one-hot encoded columns with the rest of the DataFrame, dropping the original one-hot columns.
        Args:
            data (pd.DataFrame): Input DataFrame containing features to be engineered.
        Returns:
            pd.DataFrame: DataFrame with engineered features, ready for modeling.
        """

        print(f"Extracting features")
        
        # Map 'contract' column values to integers
        data['contract'] = data['contract'].map({'monthly': 0, 
                                                 'one year': 12, 
                                                 'two year': 24})
        
        # Label Encoding
        for col in self.label_columns:
            if col in data.columns:
                data[col] = self.label_encoder.fit_transform(data[col])
        
        # One-Hot Encoding
        one_hot_data = self.one_hot_encoder.fit_transform(data[self.one_hot_columns])
        one_hot_df = pd.DataFrame(
                    one_hot_data, 
                    columns=self.one_hot_encoder.get_feature_names_out(self.one_hot_columns),
                    index=data.index
                )

        df_final = pd.concat(
                    [data.drop(self.one_hot_columns, axis=1), one_hot_df], 
                    axis=1
                )
        
        return df_final

