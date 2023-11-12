import pandas as pd
import src.embedder as e

def embed_articles(path: str) -> dict:
    """
    Read a CSV file from the given path and generate embeddings for each article.

    The function reads a CSV file containing article data, creates a summary string for each article,
    and then uses an embedding function to generate embeddings for these summary strings. The embeddings
    are stored in a dictionary keyed by the article ID.

    Parameters:
    path (str): The file path to the CSV file containing the article data.

    Returns:
    dict: A dictionary where each key is an article ID and each value is the corresponding embedding.
    """
    df = pd.read_csv(path)

    df['summary_string'] = ('product: ' + df['product_type_name'].astype(str) +
                            ' with style: ' + df['graphical_appearance_name'].astype(str) +
                            ' with color: ' + df['colour_group_name'].astype(str) +
                            ' with detail: ' + df['detail_desc'].astype(str))

    dict_of_embeddings = {row['article_id']: e.embed_string(row['summary_string']) 
                          for _, row in df.iterrows()}

    return dict_of_embeddings
