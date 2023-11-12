import os
import json
from dotenv import load_dotenv
from langchain import HuggingFaceHub

REPO_ID = "mistralai/Mistral-7B-Instruct-v0.1"
TOKEN_ENV_VAR = 'HUGGINGFACEHUB_API_TOKEN'
MODEL_KWARGS = {'temperature': 0.2, 'top_p': 0.7, 'top_k': 55}

load_dotenv()

def load_model() -> HuggingFaceHub:
    """Load the HuggingFaceHub model with the required API token and model parameters."""
    api_token = os.getenv(TOKEN_ENV_VAR)
    if not api_token:
        raise ValueError("API token is not set in the environment variables.")
    return HuggingFaceHub(repo_id=REPO_ID, huggingfacehub_api_token=api_token, model_kwargs=MODEL_KWARGS)

def summarise_user_input_json(image_caption: str, text_input: str) -> str:
    """
    Generate a summary based on user input and image caption.
    """
    prompt = (
        "[INST]\n"
        "When a user uploads an image with a caption and provides their intent in text form, "
        "your task is to discern whether they are seeking items similar to the one in the image "
        "or complementary items to pair with it. Based on the user's direction, offer five clothing "
        "items that either match or coordinate with the described item. Present your recommendations "
        "in a structured and consistent JSON format, detailing the color, style, and product type for each item.\n\n"
        f"- Assess the information given:\n    Image caption: {image_caption}\n    User's intent: {text_input}\n\n"
        "- The user's intent can express a preference for items similar to what's depicted in the image "
        "(e.g., 'black t-shirt') or for complementary items that would pair well with it "
        "(e.g., 'looking for accessories to go with this t-shirt').\n"
        "- Determine if the user's intent suggests a 'Similar' or 'Complementary' approach.\n"
        "- Include an ideal color, product type (e.g., 't-shirt', 'earring'), style, and any other relevant details "
        "in your recommendation.\n"
        "- Provide five recommendations in a JSON format. Each recommendation should be an object within an array. "
        "Ensure each object follows this consistent structure:\n"
        "    {\n"
        "        \"product_type\": \"Type of the product\",\n"
        "        \"color\": \"Color of the product\",\n"
        "        \"style\": \"Style of the product\"\n"
        "    }\n"
        "- It is crucial that the format of each recommendation remains consistent for ease of parsing. "
        "Avoid adding additional text or descriptions outside the JSON structure.\n"
        "[/INST]\n"
    )
    llm = load_model()
    user_intent = llm(prompt, max_new_tokens=2048, repetition_penalty=1.2)
    return user_intent

def parse_json_response_llm(input_string: str) -> list:
    """
    Parse the JSON response from the language model.
    """
    parsed_data = json.loads(input_string)
    return parsed_data.get("recommendations", [])
