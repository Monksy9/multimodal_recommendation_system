import src.summarise as s
import src.embedder as e
import src.distance as d
import pandas as pd

def get_recommendations(caption: str, user_input: str) -> list:
    """
    Generate recommendations based on the caption and user input.
    """
    summarised_intent_json = s.summarise_user_input_json(caption, user_input)
    print("Top 5 items from intent and image are: ")
    print(summarised_intent_json)
    return s.parse_json_response_llm(summarised_intent_json)

def process_recommendations(parsed_intent_list: list, df: pd.DataFrame, catalogue_embeddings: dict) -> pd.DataFrame:
    """
    Process the list of parsed intents and return a DataFrame with recommendations.
    """
    list_of_dfs = []
    for recommendation in parsed_intent_list:
        recommendation_embedding = e.generate_embedding(recommendation)
        distances = pd.Series({key: d.calculate_embedding_distance(value, recommendation_embedding) 
                               for key, value in catalogue_embeddings.items()}, 
                              name='distance').sort_values().head(3)

        s_df = distances.reset_index().rename(columns={'index': 'article_id'})
        merged_df = s_df.merge(df, on='article_id', how='left')
        list_of_dfs.append(merged_df)
    return pd.concat(list_of_dfs, ignore_index=True)
