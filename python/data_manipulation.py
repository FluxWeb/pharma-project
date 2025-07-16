import pandas as pd
import numpy as np


moses_data_path = "../data/moses/dataset_v1.csv"
moses_data_train_path = "../data/moses/train.csv"
moses_data_test_path = "../data/moses/test.csv"


def load_data(file_path):    
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """

    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def unique_smiles_count(df):
    """
    Count unique SMILES strings in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    
    Returns:
    int: Number of unique SMILES strings.
    """
    
    if 'SMILES' in df.columns:
        return df['SMILES'].nunique()
    else:
        print("Column 'SMILES' not found in DataFrame.")
        return 0


def non_unique_smiles_count(df):
    """
    Count non-unique SMILES strings in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    
    Returns:
    int: Number of non-unique SMILES strings.
    """
    
    if 'SMILES' in df.columns:
        return df['SMILES'].shape[0] - df['SMILES'].nunique()
    else:
        print("Column 'SMILES' not found in DataFrame.")
        return 0






if __name__ == "__main__":

    # Load the MOSES dataset
    print("Loading MOSES dataset...")
    moses = load_data(moses_data_path)
    moses_train = load_data(moses_data_train_path)
    moses_test = load_data(moses_data_test_path)

    print("HEAD MOSES DATASET\n"+moses.head())
    print("MOSES nujmber train test\n")
    print(moses["SPLIT"].value_counts())
    print("Unique SMILES in MOSES dataset: ", unique_smiles_count(moses))
    print("Non unique SMILES in MOSES dataset: ", non_unique_smiles_count(moses))

    print("Unique SMILES in MOSES train: ", unique_smiles_count(moses_train))
    print("Non unique SMILES in MOSES train: ", non_unique_smiles_count(moses_train))

    print("Unique SMILES in MOSES dataset: ", unique_smiles_count(moses_test))
    print("Non unique SMILES in MOSES dataset: ", non_unique_smiles_count(moses_test))

    print("moses SMILES object types: ",moses['SMILES'].apply(type).value_counts())
    print("Type moses: ", moses.dtypes)
    
    print(moses["SMILES"][0])

    print("max SMILES length: ", moses['SMILES'].apply(len).max())