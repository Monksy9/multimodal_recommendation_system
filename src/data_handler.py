import pandas as pd
import pickle
import os
from typing import Any, Dict

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(filepath)

def save_embeddings(embeddings: Dict[Any, Any], filepath: str) -> None:
    """
    Save embeddings to a file using pickle.

    Parameters:
    embeddings (Dict[Any, Any]): The embeddings to be saved.
    filepath (str): The path where the embeddings will be saved.
    """
    with open(filepath, 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_embeddings(filepath: str) -> Dict[Any, Any]:
    """
    Load embeddings from a pickle file.

    Parameters:
    filepath (str): The path to the pickle file containing the embeddings.

    Returns:
    Dict[Any, Any]: The loaded embeddings.
    """
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def check_embeddings_exist(filepath: str) -> bool:
    """
    Check if the embeddings file exists at the given filepath.

    Parameters:
    filepath (str): The path to the embeddings file.

    Returns:
    bool: True if the file exists, False otherwise.
    """
    return os.path.exists(filepath)
