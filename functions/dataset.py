### Imports ###
import pandas as pd
from .config import PROCESSED_DATA_DIR



class CleanDataset:
    """
    A class to clean and preprocess a dataset loaded from a CSV file.
    Args:
        input_path (Path): The file path to the input CSV dataset.
    Methods:
        clean() -> pd.DataFrame:
            Reads the CSV file, standardizes column names and string values to lowercase,
            drops the 'customerid' column, converts 'totalcharges' to numeric (setting invalid
            entries to NaN and filling with 0 where 'tenure' is 0), maps 'seniorcitizen' values
            from 0/1 to 'no'/'yes', and standardizes values in 'paymentmethod' and 'contract'
            columns. Returns the cleaned DataFrame.
    """
    def __init__(self,):
        pass

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset loaded from the specified input path.
        This method performs the following operations:
        - Reads the dataset from a CSV file.
        - Converts all column names to lowercase.
        - Converts all string values in the DataFrame to lowercase.
        - Drops the 'customerid' column.
        - Converts the 'totalcharges' column to numeric, coercing errors to NaN.
        - Sets 'totalcharges' to 0 where it is NaN and 'tenure' is 0.
        - Maps the 'seniorcitizen' column from 0/1 to 'no'/'yes'.
        - Simplifies the 'paymentmethod' column by replacing specific values.
        - Simplifies the 'contract' column by replacing "month-to-month" with "monthly".
        Returns:
            pd.DataFrame: The cleaned dataset.
        Raises:
            Exception: If any error occurs during the cleaning process.
        """

        
        try:
            raw_data = data.copy()
            
            raw_data.columns = raw_data.columns.str.lower()
            
            raw_data = raw_data.applymap(lambda s: s.lower() if type(s) == str else s)   #type: ignore
            
            raw_data = raw_data.drop(['customerid'], axis=1)
            
            raw_data['totalcharges'] = pd.to_numeric(raw_data['totalcharges'], errors='coerce')
            mask = (raw_data['totalcharges'].isna()) & (raw_data['tenure'] == 0)
            raw_data.loc[mask, 'totalcharges'] = 0
            
            raw_data['seniorcitizen'] = raw_data['seniorcitizen'].map({0: 'no', 1: 'yes'})
            
            raw_data['paymentmethod'] = raw_data['paymentmethod'].replace({
                "bank transfer (automatic)": "bank transfer",
                "credit card (automatic)": "credit card"
                })
            
            raw_data['contract'] = raw_data['contract'].replace({
                "month-to-month": "monthly"
            })
            
            # Save the cleaned dataset to the processed data directory
            processed_path = PROCESSED_DATA_DIR / f"churn_clean_data.csv"
            raw_data.to_csv(processed_path, index=False)
            
            return raw_data
        
        except Exception as e:
            print(f"Error cleaning Dataset: {e}")
            raise
        
