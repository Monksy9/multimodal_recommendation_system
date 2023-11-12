from sentence_transformers import SentenceTransformer
import src.distance as d

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def embed_string(sentence: str):
    embeddings = model.encode([sentence])
    return embeddings

def combine_recommendation(parsed_intent_dict: dict) -> str:
    return f"A {parsed_intent_dict['color']} {parsed_intent_dict['product_type']} of style {parsed_intent_dict['style']}"

def generate_embedding(recommendation: dict):
    return embed_string(combine_recommendation(recommendation))

def calculate_distances(catalogue_embeddings: dict, recommendation_embedding):
    return {key: d.calculate_distance(value, recommendation_embedding) for key, value in catalogue_embeddings.items()}
