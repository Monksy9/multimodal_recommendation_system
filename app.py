import src.config as c
from src.data_handler import load_data, save_embeddings, load_embeddings, check_embeddings_exist
from src.image_captioning import ImageCaptioner
from src.recommendation import get_recommendations, process_recommendations
import yaml

TOP_N_RECOMMENDATIONS = 5
IMG_PATH, STRING, ARTICLES_CSV, EMBEDDINGS_PICKLE = c.load_config()

def main():
    df = load_data(ARTICLES_CSV)
    captioner = ImageCaptioner()
    caption = captioner.get_image_caption(IMG_PATH)
    parsed_intent_list = get_recommendations(caption, STRING)

    if check_embeddings_exist(EMBEDDINGS_PICKLE):
        print("Reading cached embeddings of catalogue, as precalculated.")
        catalogue_embeddings = load_embeddings(EMBEDDINGS_PICKLE)
    else:
        print("Re-calculating embeddings:")
        catalogue_embeddings = ec.embed_articles(path=ARTICLES_CSV)
        save_embeddings(catalogue_embeddings, EMBEDDINGS_PICKLE)

    combined_df = process_recommendations(parsed_intent_list, df, catalogue_embeddings)
    
    combined_df.sort_values('distance').head(TOP_N_RECOMMENDATIONS).to_csv('top_5_recommendations.csv')
    
    print("Top 5 recommendations saved in top_5_recommendations.csv")

if __name__ == "__main__":
    main()
