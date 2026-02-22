import requests
import json

def emotion_detector(text_to_analyze):
    """
    Detects emotions in a given text using Watson NLP EmotionPredict API.

    Parameters:
        text_to_analyze (str): Text input from the user.

    Returns:
        dict: Dictionary containing scores for each emotion and dominant emotion.
              If input is blank, returns None for all values.
    """

    # API endpoint for Watson NLP EmotionPredict
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"

    # Required headers for API call
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    # Prepare JSON payload with user input
    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    # Send POST request to Watson NLP API
    response = requests.post(url, headers=headers, json=payload)

    # --------- Error handling for blank input ---------
    if response.status_code == 400:
        # If user input is empty, return dictionary with None values
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    # --------- Parse API response ---------
    try:
        response_dict = response.json()  # Convert response to dictionary
        emotions = response_dict["emotionPredictions"][0]["emotion"]
    except (KeyError, IndexError, json.JSONDecodeError):
        # If response format is unexpected, return None values
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    # --------- Extract individual emotion scores ---------
    anger = emotions.get("anger")
    disgust = emotions.get("disgust")
    fear = emotions.get("fear")
    joy = emotions.get("joy")
    sadness = emotions.get("sadness")

    # --------- Determine dominant emotion ---------
    emotion_scores = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness
    }

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    # --------- Return formatted dictionary ---------
    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion
    }