import numpy as np
from typing import List

def calculate_embedding_distance(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate the Euclidean distance between two embeddings.
    Parameters:
    embedding1 (List[float]): The first embedding vector.
    embedding2 (List[float]): The second embedding vector.

    Returns:
    float: The Euclidean distance between the two embeddings.
    """
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance