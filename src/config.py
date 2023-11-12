import yaml
from typing import Tuple

def load_config() -> Tuple[str, str, str, str]:
    """
    Load configuration from a YAML file.

    This function reads configuration parameters from 'config.yaml' and returns them.

    Returns:
    Tuple[str, str, str, str]: A tuple containing the image path, user input string,
                                articles CSV file path, and embeddings pickle file path.
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    img_path = config['img_path']
    user_input = config['user_input']
    articles_csv = config['articles_csv']
    embeddings_pickle = config['embeddings_pickle']
    
    return img_path, user_input, articles_csv, embeddings_pickle
