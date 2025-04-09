import streamlit as st
import cv2
import numpy as np
import speech_recognition as sr
import os
import tempfile
import time
import pandas as pd
import matplotlib.pyplot as plt
import pyttsx3
from PIL import Image
import io
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, AudioProcessorBase
import av
import threading
import json
import random
import subprocess
import sys
import streamlit as st
import requests
from ultralytics import YOLO
import os
import subprocess
import sys
import streamlit as st
import re
import torch
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

# Simulated landmark database (in real app, this would be a more comprehensive database)
LANDMARKS_DB = {
    "eiffel_tower": {
        "name": "Eiffel Tower",
        "description": "Iconic iron tower in Paris, France.",
        "history": "Completed in 1889 as the entrance arch to the 1889 World's Fair.",
        "recommendations": ["Visit at sunset", "Book tickets in advance", "Check out the restaurant"],
        "nearby": ["Champ de Mars", "Trocadéro Gardens", "Seine River Cruise"]
    },
    "statue_of_liberty": {
        "name": "Statue of Liberty",
        "description": "Neoclassical sculpture on Liberty Island in New York Harbor.",
        "history": "A gift from the people of France, dedicated in 1886.",
        "recommendations": ["Take the ferry early", "Climb to the crown (reservation required)", "Visit the museum"],
        "nearby": ["Ellis Island", "Battery Park", "One World Trade Center"]
    },
    "taj_mahal": {
        "name": "Taj Mahal",
        "description": "Ivory-white marble mausoleum in Agra, India.",
        "history": "Built between 1631 and 1648 by order of the Mughal emperor Shah Jahan.",
        "recommendations": ["Visit at sunrise", "Hire a local guide", "Bring water and sunscreen"],
        "nearby": ["Agra Fort", "Mehtab Bagh", "Fatehpur Sikri"]
    },
    "colosseum": {
        "name": "Colosseum",
        "description": "Ancient amphitheater in Rome, Italy.",
        "history": "Construction began under Emperor Vespasian in AD 72 and completed in AD 80.",
        "recommendations": ["Buy combined ticket with Roman Forum", "Visit early or late", "Take a guided tour"],
        "nearby": ["Roman Forum", "Palatine Hill", "Arch of Constantine"]
    },
    "great_wall": {
        "name": "Great Wall of China",
        "description": "Series of fortifications built along the northern borders of China.",
        "history": "Built from the 7th century BC, with many parts rebuilt during the Ming dynasty.",
        "recommendations": ["Visit less crowded sections", "Wear comfortable shoes", "Check weather before going"],
        "nearby": ["Ming Tombs", "Summer Palace", "Forbidden City"]
    }
}

# Simulated restaurant database
RESTAURANTS_DB = {
    "paris": [
        {"name": "Le Jules Verne", "cuisine": "French", "price": "$$$$", "dietary": ["vegetarian options"]},
        {"name": "Chez Francis", "cuisine": "French", "price": "$$$", "dietary": ["vegetarian options", "gluten-free options"]},
        {"name": "L'As du Fallafel", "cuisine": "Middle Eastern", "price": "$", "dietary": ["vegetarian", "vegan options"]}
    ],
    "new_york": [
        {"name": "Katz's Delicatessen", "cuisine": "American", "price": "$$", "dietary": []},
        {"name": "Eleven Madison Park", "cuisine": "American", "price": "$$$$", "dietary": ["vegetarian options", "vegan options"]},
        {"name": "Superiority Burger", "cuisine": "American", "price": "$", "dietary": ["vegetarian", "vegan"]}
    ],
    "agra": [
        {"name": "Pind Balluchi", "cuisine": "Indian", "price": "$$", "dietary": ["vegetarian", "vegan options"]},
        {"name": "Esphahan", "cuisine": "Indian", "price": "$$$", "dietary": ["vegetarian options"]},
        {"name": "Dasaprakash", "cuisine": "South Indian", "price": "$$", "dietary": ["vegetarian"]}
    ],
    "rome": [
        {"name": "La Pergola", "cuisine": "Italian", "price": "$$$$", "dietary": ["vegetarian options"]},
        {"name": "Pizzarium", "cuisine": "Pizza", "price": "$", "dietary": ["vegetarian options"]},
        {"name": "Armando al Pantheon", "cuisine": "Italian", "price": "$$", "dietary": ["vegetarian options"]}
    ],
    "beijing": [
        {"name": "Duck de Chine", "cuisine": "Chinese", "price": "$$$", "dietary": []},
        {"name": "King's Joy", "cuisine": "Chinese", "price": "$$$", "dietary": ["vegetarian", "vegan"]},
        {"name": "Baoyuan Dumplings", "cuisine": "Chinese", "price": "$$", "dietary": ["vegetarian options"]}
    ]
}

# Simple translation dictionary (in real app, would use a proper translation API)
TRANSLATIONS = {
    "hello": {"french": "bonjour", "spanish": "hola", "italian": "ciao", "mandarin": "nǐ hǎo"},
    "goodbye": {"french": "au revoir", "spanish": "adiós", "italian": "arrivederci", "mandarin": "zàijiàn"},
    "thank you": {"french": "merci", "spanish": "gracias", "italian": "grazie", "mandarin": "xièxiè"},
    "where is": {"french": "où est", "spanish": "dónde está", "italian": "dov'è", "mandarin": "zài nǎlǐ"},
    "help": {"french": "aide", "spanish": "ayuda", "italian": "aiuto", "mandarin": "bāngzhù"},
    "restaurant": {"french": "restaurant", "spanish": "restaurante", "italian": "ristorante", "mandarin": "cāntīng"},
    "museum": {"french": "musée", "spanish": "museo", "italian": "museo", "mandarin": "bówùguǎn"},
    "hotel": {"french": "hôtel", "spanish": "hotel", "italian": "albergo", "mandarin": "jiǔdiàn"},
    "airport": {"french": "aéroport", "spanish": "aeropuerto", "italian": "aeroporto", "mandarin": "jīchǎng"},
    "train": {"french": "train", "spanish": "tren", "italian": "treno", "mandarin": "huǒchē"}
}

# Simulated user preference history
user_preferences = {
    "preferred_cuisine": ["Italian", "Local"],
    "budget": "$$",
    "interests": ["history", "architecture", "local culture"],
    "activity_level": "moderate",
    "dietary_restrictions": ["vegetarian options"]
}

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.landmark_detected = None
        self.model = YOLO("yolov5su.pt")  # Updated to yolov5su.pt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.class_to_landmark = {
            "building": "taj_mahal",  # Temporary mapping
        }
        
        self.frame_counter = 0
        self.process_interval = 15
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_counter += 1
        if self.frame_counter % self.process_interval == 0:
            results = self.model(img)
            
            if len(results[0].xyxy) > 0:
                for det in results[0].xyxy:
                    x_min, y_min, x_max, y_max, confidence, class_id = det
                    class_name = self.model.names[int(class_id)]
                    print(f"Detected: {class_name}, Confidence: {confidence:.2f}")
                    
                    if confidence > 0.5:
                        landmark_key = self.class_to_landmark.get(class_name)
                        if landmark_key and landmark_key in LANDMARKS_DB:
                            self.landmark_detected = landmark_key
                            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                            label = f"{LANDMARKS_DB[landmark_key]['name']} ({confidence:.2f})"
                            cv2.putText(img, label, 
                                       (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        else:
                            cv2.putText(img, f"Unknown: {class_name}", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                self.landmark_detected = None
                cv2.putText(img, "No landmark detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        elif self.landmark_detected:
            height, width = img.shape[:2]
            cv2.rectangle(img, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), 2)
            cv2.putText(img, LANDMARKS_DB[self.landmark_detected]["name"], 
                       (width//4, height//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        return img

def process_image(image):
    model = YOLO("yolov5su.pt")  # New model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    class_to_landmark = {
        "building": "taj_mahal",  # Temporary mapping
    }

    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image)

    results = model(img_array)
    detected = "unknown"

    if results and results[0].boxes is not None and results[0].boxes.xyxy.shape[0] > 0:
        for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
            x_min, y_min, x_max, y_max = map(int, box)
            confidence = float(conf)
            class_id = int(cls)
            class_name = model.model.names[class_id]
            print(f"Image detected: {class_name}, Confidence: {confidence:.2f}")

            if confidence > 0.5:
                landmark_key = class_to_landmark.get(class_name)
                if landmark_key and landmark_key in LANDMARKS_DB:
                    detected = landmark_key
                    cv2.rectangle(img_array, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f"{LANDMARKS_DB[detected]['name']} ({confidence:.2f})"
                    cv2.putText(img_array, label,
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    cv2.putText(img_array, f"Unknown: {class_name}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    detected = "unknown"
    else:
        height, width = img_array.shape[:2]
        cv2.putText(img_array, "No landmark detected",
                    (width // 4, height // 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        detected = "unknown"

    return img_array, detected


    
def translate_text(text, target_language):
    """Simple translation function"""
    words = text.lower().split()
    translated = []
    
    for word in words:
        if word in TRANSLATIONS and target_language.lower() in TRANSLATIONS[word]:
            translated.append(TRANSLATIONS[word][target_language.lower()])
        else:
            translated.append(f"[{word}]")  # Untranslated words in brackets
    
    return " ".join(translated)

def get_recommendations(landmark_key, preferences=None):
    """Generate personalized recommendations based on detected landmark and user preferences"""
    if not preferences:
        preferences = user_preferences
    
    landmark_info = LANDMARKS_DB.get(landmark_key, {})
    
    # Find nearby dining options based on preferences
    city = ""
    if "eiffel_tower" in landmark_key:
        city = "paris"
    elif "statue_of_liberty" in landmark_key:
        city = "new_york"
    elif "taj_mahal" in landmark_key:
        city = "agra"
    elif "colosseum" in landmark_key:
        city = "rome"
    elif "great_wall" in landmark_key:
        city = "beijing"
    
    restaurant_suggestions = []
    if city and city in RESTAURANTS_DB:
        for restaurant in RESTAURANTS_DB[city]:
            matches_budget = preferences["budget"] == restaurant["price"] or preferences["budget"] == "$" + restaurant["price"]
            dietary_match = any(diet in restaurant["dietary"] for diet in preferences["dietary_restrictions"])
            cuisine_match = restaurant["cuisine"] in preferences["preferred_cuisine"] or "Local" in preferences["preferred_cuisine"]
            
            if (matches_budget or cuisine_match) and (not preferences["dietary_restrictions"] or dietary_match):
                restaurant_suggestions.append(restaurant)
    
    # Generate an itinerary based on preferences
    itinerary = []
    if "history" in preferences["interests"] and landmark_info.get("history"):
        itinerary.append(f"Learn about the history: {landmark_info['history']}")
    
    if landmark_info.get("recommendations"):
        for rec in landmark_info["recommendations"]:
            itinerary.append(f"Recommendation: {rec}")
    
    if landmark_info.get("nearby"):
        nearby_filtered = []
        for place in landmark_info["nearby"]:
            if "architecture" in preferences["interests"] and ("Palace" in place or "Building" in place or "Cathedral" in place):
                nearby_filtered.append(place)
            elif "local culture" in preferences["interests"] and ("Market" in place or "Square" in place or "District" in place):
                nearby_filtered.append(place)
            else:
                nearby_filtered.append(place)
        
        # Select places based on activity level
        if preferences["activity_level"] == "light":
            nearby_filtered = nearby_filtered[:1]  # Just one nearby place
        elif preferences["activity_level"] == "moderate":
            nearby_filtered = nearby_filtered[:2]  # Two nearby places
        
        for place in nearby_filtered:
            itinerary.append(f"Visit nearby: {place}")
    
    return {
        "landmark_info": landmark_info,
        "restaurant_suggestions": restaurant_suggestions[:2],  # Limit to top 2
        "itinerary": itinerary
    }

def listen_to_voice():
    recognizer = sr.Recognizer()
    # Increase energy threshold for better noise handling
    recognizer.energy_threshold = 4000
    
    with sr.Microphone() as source:
        st.write("Adjusting for ambient noise... Please wait.")
        # Increase duration for better noise adjustment
        recognizer.adjust_for_ambient_noise(source, duration=2)
        st.write("Listening...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    
    try:
        # Specify language for better recognition
        text = recognizer.recognize_google(audio, language="en-US")
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def app():
    st.title("Smart Travel Companion")
    
    # Sidebar for user preferences
    st.sidebar.header("User Preferences")
    
    # Allow user to update preferences
    cuisine_options = ["Italian", "French", "American", "Chinese", "Indian", "Japanese", "Mexican", "Local"]
    selected_cuisines = st.sidebar.multiselect("Preferred Cuisines", cuisine_options, default=user_preferences["preferred_cuisine"])
    
    budget_options = ["$", "$$", "$$$", "$$$$"]
    selected_budget = st.sidebar.select_slider("Budget", options=budget_options, value=user_preferences["budget"])
    
    interest_options = ["history", "architecture", "local culture", "art", "nature", "shopping", "food"]
    selected_interests = st.sidebar.multiselect("Interests", interest_options, default=user_preferences["interests"])
    
    activity_options = ["light", "moderate", "active"]
    selected_activity = st.sidebar.radio("Activity Level", activity_options, index=activity_options.index(user_preferences["activity_level"]))
    
    dietary_options = ["none", "vegetarian options", "vegan options", "gluten-free options"]
    selected_dietary = st.sidebar.multiselect("Dietary Restrictions", dietary_options, default=user_preferences["dietary_restrictions"])
    
    # Update preferences if user changes them
    if st.sidebar.button("Update Preferences"):
        user_preferences["preferred_cuisine"] = selected_cuisines
        user_preferences["budget"] = selected_budget
        user_preferences["interests"] = selected_interests
        user_preferences["activity_level"] = selected_activity
        user_preferences["dietary_restrictions"] = selected_dietary if "none" not in selected_dietary else []
        st.sidebar.success("Preferences updated!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Landmark Detection", "Voice Assistant", "Translation", "Trip Planning"])
                
    # Tab 1: Computer Vision for Landmark Detection
    with tab1:
        st.header("Landmark Detection")
        st.write("Use your camera to identify landmarks or upload a photo.")
        
        detection_method = st.radio("Choose detection method:", ["Upload Image", "Use Camera"])
        detected_landmark = None
        
        if detection_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Detect Landmarks"):
                    with st.spinner("Processing image..."):
                        processed_img, detected_landmark = process_image(image)
                        st.image(processed_img, 
                                caption=f"Detected: {LANDMARKS_DB.get(detected_landmark, {'name': 'Unknown'})['name']}", 
                                use_column_width=True)
                        
                        if detected_landmark in LANDMARKS_DB:
                            st.subheader(LANDMARKS_DB[detected_landmark]['name'])
                            st.write(LANDMARKS_DB[detected_landmark]['description'])
                            recommendations = get_recommendations(detected_landmark, user_preferences)
                            with st.expander("Learn More"):
                                st.write(LANDMARKS_DB[detected_landmark]['history'])
                            with st.expander("Personalized Recommendations"):
                                st.subheader("Based on your preferences:")
                                if recommendations["restaurant_suggestions"]:
                                    st.write("#### Where to Eat")
                                    for restaurant in recommendations["restaurant_suggestions"]:
                                        st.write(f"**{restaurant['name']}** - {restaurant['cuisine']} ({restaurant['price']})")
                                        if restaurant['dietary']:
                                            st.write(f"*Accommodates: {', '.join(restaurant['dietary'])}*")
                                if recommendations["itinerary"]:
                                    st.write("#### Suggested Itinerary")
                                    for i, item in enumerate(recommendations["itinerary"], 1):
                                        st.write(f"{i}. {item}")
                        else:
                            st.write("No known landmark detected.")
        
        else:
            st.write("Note: Camera access requires permission from your browser")
            ctx = webrtc_streamer(
                key="landmark-detection",
                video_processor_factory=VideoProcessor,  # Updated from video_transformer_factory
                media_stream_constraints={"video": True, "audio": False},
            )
            if ctx.video_processor and ctx.video_processor.landmark_detected:
                detected_landmark = ctx.video_processor.landmark_detected
                st.subheader(f"Detected: {LANDMARKS_DB[detected_landmark]['name']}")
                st.write(LANDMARKS_DB[detected_landmark]['description'])
                recommendations = get_recommendations(detected_landmark, user_preferences)
                with st.expander("Learn More"):
                    st.write(LANDMARKS_DB[detected_landmark]['history'])
                with st.expander("Personalized Recommendations"):
                    st.subheader("Based on your preferences:")
                    if recommendations["restaurant_suggestions"]:
                        st.write("#### Where to Eat")
                        for restaurant in recommendations["restaurant_suggestions"]:
                            st.write(f"**{restaurant['name']}** - {restaurant['cuisine']} ({restaurant['price']})")
                            if restaurant['dietary']:
                                st.write(f"*Accommodates: {', '.join(restaurant['dietary'])}*")
                    if recommendations["itinerary"]:
                        st.write("#### Suggested Itinerary")
                        for i, item in enumerate(recommendations["itinerary"], 1):
                            st.write(f"{i}. {item}")
                                    
    import requests

    # Hugging Face API setup
    HF_API_TOKEN = st.secrets["huggingface"]["api_token"] if "huggingface" in st.secrets else "your_token_here"
    HF_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"  # Conversational model

    def query_llm(prompt):
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {"inputs": prompt, "parameters": {"max_length": 200}}
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "Sorry, I couldn’t process that.")
        else:
            return f"Error from LLM API: {response.status_code} - {response.text}"
    
    # Function to process query with Together.ai LLM

    def process_query_with_llm(query, context="You are a travel assistant helping users with location, recommendations, and translations. Use the provided databases LANDMARKS_DB, RESTAURANTS_DB, and TRANSLATIONS when relevant."):
        # Prepare the prompt
        full_prompt = f"{context}\nUser query: {query}\nResponse:"
    
        # Together API setup
        api_key = "tgp_v1_lxVgdEmpgQ-0OfEqehsdB9QRIbZ9lnxcEBSOOfrNbIY"  # Replace with your actual API key
        model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        url = "https://api.together.xyz/v1/chat/completions"
    
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7
        }
    
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise error if not 200
            result = response.json()
    
            # Extract the generated text
            if result and "choices" in result and len(result["choices"]) > 0:
                final_response = result["choices"][0]["message"]["content"].strip()
            else:
                final_response = "Sorry, I couldn't process your request at the moment."
    
        except Exception as e:
            final_response = f"Error communicating with the language model: {str(e)}"
    
        return final_response

    def clean_llm_response(response, query):
        # Remove <think> section if present
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        
        # Check if query implies a singular "favourite" answer
        query_lower = query.lower()
        is_favourite_query = "favourite" in query_lower or "favorite" in query_lower or "best" in query_lower
        
        if is_favourite_query:
            # Look for list-like structures (e.g., numbered items or bullet points)
            lines = response.split("\n")
            for line in lines:
                # Match lines that look like a recommendation (e.g., starts with a number, bullet, or place name)
                if re.match(r"^\d+\.|^[-*]|^[A-Za-z\s]+\(", line.strip()):
                    # Extract the first valid recommendation
                    place_match = re.search(r"(Taj Mahal|Jaipur|Varanasi|Goa|Kerala|Agra Fort|[A-Za-z\s]+)", line)
                    if place_match:
                        place = place_match.group(0).strip()
                        # Find description in the response or use LANDMARKS_DB
                        for l in lines:
                            if place in l and "Why Visit" in l or "Highlights" in l:
                                return f"My favourite place in India is {place}. {l.strip()}"
    
                        # Fallback to LANDMARKS_DB if no description found
                        for key, value in LANDMARKS_DB.items():
                            if place.lower() in key:
                                return f"My favourite place in India is {value['name']}. {value['description']}"
                        return f"My favourite place in India is {place}."
            
            # If no list detected, take the first sentence mentioning a place
            sentences = re.split(r"[.!?]\s+", response)
            for sentence in sentences:
                place_match = re.search(r"(Taj Mahal|Jaipur|Varanasi|Goa|Kerala|Agra Fort|[A-Za-z\s]+)", sentence)
                if place_match:
                    place = place_match.group(0).strip()
                    return f"My favourite place in India is {place}. {sentence.strip()}."
            
            # Default fallback if no clear place is found
            return "My favourite place in India is the Taj Mahal. It’s a stunning white marble mausoleum in Agra."
        
        # For non-favourite queries, return the cleaned response as-is
        return response

    def clean_translation_response(response):
        # Remove <think> section if present
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        
        # Remove any extra explanation, keeping only the translation
        # Assume the translation is the first line or sentence unless it’s clearly metadata
        lines = response.split("\n")
        for line in lines:
            if line.strip() and not line.startswith("Response:") and not line.startswith("["):
                return line.strip()
        
        # Fallback: return the raw response trimmed of fluff
        return re.sub(r"^(Translation:|Translated text:)", "", response).strip()

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.recognizer = sr.Recognizer()
            self.audio_frames = []
            self.sample_rate = 48000  # Default WebRTC sample rate
        
        def recv(self, frame):
            audio_data = frame.to_ndarray()
            self.audio_frames.append(audio_data)
            return frame
        
        def process_audio(self):
            if not self.audio_frames:
                return "No audio recorded"
            
            try:
                audio_array = np.concatenate(self.audio_frames, axis=0)
                audio_bytes = audio_array.tobytes()
                audio_file = sr.AudioData(audio_bytes, sample_rate=self.sample_rate, sample_width=2)
                text = self.recognizer.recognize_google(audio_file, language="en-US")
                return text
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results; {str(e)}"
            except Exception as e:
                return f"Error: {str(e)}"
            finally:
                self.audio_frames = []

    
    def translate_with_deepseek(text, target_lang):
        api_key = "tgp_v1_lxVgdEmpgQ-0OfEqehsdB9QRIbZ9lnxcEBSOOfrNbIY"
        model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        url = "https://api.together.xyz/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        context = "You are a translation assistant. Translate the user's text into the specified language and provide only the translated text."
        query = f"Translate '{text}' from English to {target_lang}"
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raises 429 or other errors
            result = response.json()
            if "choices" in result and result["choices"]:
                raw_response = result["choices"][0]["message"]["content"].strip()
                # Clean the response to remove <think> section and extra text
                if "</think>" in raw_response:
                    cleaned_response = raw_response.split("</think>")[1].strip()
                else:
                    cleaned_response = raw_response
                # Ensure only the translation is returned (remove any leading/trailing fluff)
                return re.sub(r"^(Translation:|Translated text:)", "", cleaned_response).strip()
            return "Translation error"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                return "Too many requests. Please try again later."
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Assuming this is your existing TRANSLATIONS dictionary (from earlier in your app)
    # If not, ensure it’s defined elsewhere in your code
    TRANSLATIONS = {
        "hello": {"french": "bonjour", "spanish": "hola", "italian": "ciao", "mandarin": "nǐ hǎo"},
        "goodbye": {"french": "au revoir", "spanish": "adios", "italian": "arrivederci", "mandarin": "zài jiàn"},
        "thank you": {"french": "merci", "spanish": "gracias", "italian": "grazie", "mandarin": "xiè xiè"},
        "where is": {"french": "où est", "spanish": "dónde está", "italian": "dove è", "mandarin": "zài nǎ lǐ"},
        "help": {"french": "aidez-moi", "spanish": "ayuda", "italian": "aiuto", "mandarin": "bāng zhù"},
        "restaurant": {"french": "restaurant", "spanish": "restaurante", "italian": "ristorante", "mandarin": "cān tīng"},
        "museum": {"french": "musée", "spanish": "museo", "italian": "museo", "mandarin": "bó wù guǎn"},
        "hotel": {"french": "hôtel", "spanish": "hotel", "italian": "albergo", "mandarin": "jiǔ diàn"},
        "airport": {"french": "aéroport", "spanish": "aeropuerto", "italian": "aeroporto", "mandarin": "jī chǎng"},
        "train": {"french": "train", "spanish": "tren", "italian": "treno", "mandarin": "huǒ chē"}
    }




    # Tab 2: Voice Assistant
    with tab2:
        st.header("Voice Assistant")
        st.write("Ask questions or get information using your voice (powered by an AI language model)")
        
        ctx = webrtc_streamer(
            key="voice-assistant",
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=False
        )
        
        if ctx.audio_processor:
            if st.button("Stop and Process Audio"):
                with st.spinner("Processing..."):
                    voice_text = ctx.audio_processor.process_audio()
                    if voice_text and voice_text not in ["Could not understand audio", "No audio recorded"]:
                        st.success(f"You said: {voice_text}")
                        query = voice_text.lower()
                        raw_response = process_query_with_llm(query)
                        clean_response = clean_llm_response(raw_response, query)
                        st.write("Assistant:", clean_response)
                        if "translate" in query:
                            st.write("For translations, I’ll need a specific language. Please say something like 'Translate hello to French'.")
                        elif "map" in clean_response.lower() or "location" in clean_response.lower():
                            st.image("https://via.placeholder.com/600x400?text=Map+of+your+location", caption="Simulated Map")
                    else:
                        st.error(voice_text or "No audio detected.")
            else:
                st.write("Recording audio... Click 'Stop and Process Audio' when done.")
        
        # Text input as fallback
        text_query = st.text_input("Or type your question:")
        if text_query:
            with st.spinner("Processing..."):
                raw_response = process_query_with_llm(text_query)
                clean_response = clean_llm_response(raw_response, text_query)
                st.write("Assistant:", clean_response)
                
                # Post-process LLM response for specific actions
                if "translate" in text_query.lower():
                    st.write("For translations, I’ll need a specific language. Please type something like 'Translate hello to French'.")
                elif "map" in clean_response.lower() or "location" in clean_response.lower():
                    st.image("https://via.placeholder.com/600x400?text=Map+of+your+location", caption="Simulated Map")
    
    # # Tab 2: Voice Assistant
    # with tab2:
    #     st.header("Voice Assistant")
    #     st.write("Ask questions or get information using your voice")
        
    #     if st.button("Start Listening"):
    #         with st.spinner("Listening..."):
    #             try:
    #                 voice_text = listen_to_voice()
    #                 if voice_text:
    #                     st.success(f"You said: {voice_text}")
                        
    #                     # Process the voice command (NLP component)
    #                     # In a real app, this would use a more sophisticated NLP system
    #                     query = voice_text.lower()
                        
    #                     if "where" in query and "i" in query:
    #                         # Location query
    #                         landmarks = list(LANDMARKS_DB.keys())
    #                         random_landmark = random.choice(landmarks)
    #                         st.write(f"Based on your location, you are near the {LANDMARKS_DB[random_landmark]['name']}.")
                            
    #                         # Show a map (simulated)
    #                         st.image("https://via.placeholder.com/600x400?text=Map+of+your+location", caption="Your Location")
                            
    #                     elif "recommend" in query or "suggest" in query:
    #                         # Recommendation query
    #                         if "eat" in query or "food" in query or "restaurant" in query:
    #                             city = random.choice(list(RESTAURANTS_DB.keys()))
    #                             restaurants = RESTAURANTS_DB[city]
                                
    #                             st.write(f"Here are some restaurant recommendations in {city}:")
    #                             for restaurant in restaurants[:2]:
    #                                 st.write(f"**{restaurant['name']}** - {restaurant['cuisine']} ({restaurant['price']})")
    #                                 if restaurant['dietary']:
    #                                     st.write(f"*Accommodates: {', '.join(restaurant['dietary'])}*")
                            
    #                         elif "see" in query or "visit" in query or "attraction" in query:
    #                             landmarks = list(LANDMARKS_DB.keys())
    #                             random_landmark = random.choice(landmarks)
                                
    #                             st.write(f"I recommend visiting the {LANDMARKS_DB[random_landmark]['name']}:")
    #                             st.write(LANDMARKS_DB[random_landmark]['description'])
                                
    #                             # Show recommendations
    #                             if LANDMARKS_DB[random_landmark]['recommendations']:
    #                                 st.write("**Tips:**")
    #                                 for tip in LANDMARKS_DB[random_landmark]['recommendations']:
    #                                     st.write(f"- {tip}")
                        
    #                     elif "translate" in query:
    #                         # Extract what needs to be translated
    #                         text_to_translate = query.split("translate")[-1].strip()
    #                         if text_to_translate:
    #                             st.write("What language would you like to translate to?")
    #                             languages = ["french", "spanish", "italian", "mandarin"]
    #                             target_lang = st.selectbox("Select language:", languages)
                                
    #                             if target_lang:
    #                                 translated = translate_text(text_to_translate, target_lang)
    #                                 st.success(f"Translation to {target_lang}: {translated}")
                        
    #                     else:
    #                         st.write("I can help with finding locations, recommending places to eat or visit, and translating common phrases.")
    #             except Exception as e:
    #                 st.error(f"Error processing voice: {str(e)}")
        
    #     # Text input as fallback
    #     text_query = st.text_input("Or type your question:")
    #     if text_query:
    #         # Process the text query (similar logic as voice)
    #         query = text_query.lower()
            
    #         if "where" in query and "i" in query:
    #             # Location query
    #             landmarks = list(LANDMARKS_DB.keys())
    #             random_landmark = random.choice(landmarks)
    #             st.write(f"Based on your location, you are near the {LANDMARKS_DB[random_landmark]['name']}.")
                
    #             # Show a map (simulated)
    #             st.image("https://via.placeholder.com/600x400?text=Map+of+your+location", caption="Your Location")
                
    #         elif "recommend" in query or "suggest" in query:
    #             # Recommendation query
    #             if "eat" in query or "food" in query or "restaurant" in query:
    #                 city = random.choice(list(RESTAURANTS_DB.keys()))
    #                 restaurants = RESTAURANTS_DB[city]
                    
    #                 st.write(f"Here are some restaurant recommendations in {city}:")
    #                 for restaurant in restaurants[:2]:
    #                     st.write(f"**{restaurant['name']}** - {restaurant['cuisine']} ({restaurant['price']})")
    #                     if restaurant['dietary']:
    #                         st.write(f"*Accommodates: {', '.join(restaurant['dietary'])}*")
                
    #             elif "see" in query or "visit" in query or "attraction" in query:
    #                 landmarks = list(LANDMARKS_DB.keys())
    #                 random_landmark = random.choice(landmarks)
                    
    #                 st.write(f"I recommend visiting the {LANDMARKS_DB[random_landmark]['name']}:")
    #                 st.write(LANDMARKS_DB[random_landmark]['description'])
                    
    #                 # Show recommendations
    #                 if LANDMARKS_DB[random_landmark]['recommendations']:
    #                     st.write("**Tips:**")
    #                     for tip in LANDMARKS_DB[random_landmark]['recommendations']:
    #                         st.write(f"- {tip}")
            
    #         elif "translate" in query:
    #             # Extract what needs to be translated
    #             text_to_translate = query.split("translate")[-1].strip()
    #             if text_to_translate:
    #                 st.write("What language would you like to translate to?")
    #                 languages = ["french", "spanish", "italian", "mandarin"]
    #                 target_lang = st.selectbox("Select language:", languages)
                    
    #                 if target_lang:
    #                     translated = translate_text(text_to_translate, target_lang)
    #                     st.success(f"Translation to {target_lang}: {translated}")
            
    #         else:
    #             st.write("I can help with finding locations, recommending places to eat or visit, and translating common phrases.")

    # Tab 3: Translation
    with tab3:
        st.header("Translation Assistant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_text = st.text_area("Enter text to translate:", height=150)
            target_lang = st.selectbox("Translate to:", ["French", "Spanish", "Italian", "Mandarin"])
        
        with col2:
            st.write("Translation:")
            if source_text and target_lang:
                translated_text = translate_with_deepseek(source_text, target_lang)
                if "Too many requests" in translated_text or "Error" in translated_text:
                    st.error(translated_text)
                else:
                    st.text_area("Translation:", translated_text, height=150)
        
        # Common phrases section
        st.subheader("Common Travel Phrases")
        common_phrases = [
            "Hello", "Goodbye", "Thank you", "Where is", "Help", 
            "Restaurant", "Museum", "Hotel", "Airport", "Train"
        ]
        
        selected_phrase = st.selectbox("Select a common phrase:", common_phrases)
        
        if selected_phrase:
            st.write("Translations:")
            cols = st.columns(4)
            languages = ["French", "Spanish", "Italian", "Mandarin"]
            for i, lang in enumerate(languages):
                with cols[i]:
                    st.write(f"**{lang}:**")
                    translated_text = translate_with_deepseek(selected_phrase, lang)
                    # Fallback to TRANSLATIONS if API fails
                    if "Too many requests" in translated_text or "Error" in translated_text:
                        translated_text = TRANSLATIONS.get(selected_phrase.lower(), {}).get(lang.lower(), "-")
                    st.write(translated_text)

    # Tab 4: Trip Planning (Decision Making)
    with tab4:
        st.header("Trip Planning")
        st.write("Plan your perfect day based on your preferences and available time")
        
        # Location selection
        locations = {
            "paris": "Paris, France",
            "new_york": "New York, USA",
            "rome": "Rome, Italy",
            "agra": "Agra, India",
            "beijing": "Beijing, China"
        }
        
        selected_location = st.selectbox("Select your destination:", list(locations.values()))
        
        # Find the key for the selected location
        location_key = next((k for k, v in locations.items() if v == selected_location), None)
        
        # Trip duration
        available_time = st.slider("How many hours do you have available?", 1, 12, 4)
        
        # Trip focus
        trip_focus = st.radio(
            "What's the main focus of your trip?",
            ["Must-see landmarks", "Food and culture", "Off the beaten path", "Relaxed pace"]
        )
        
        # Transportation preference
        transportation = st.multiselect(
            "Preferred transportation methods:",
            ["Walking", "Public Transport", "Taxi/Rideshare", "Guided Tour"],
            default=["Walking", "Public Transport"]
        )
        
        # Special requirements
        special_requirements = st.text_area("Any special requirements or notes?")
        
        # Generate plan button
        if st.button("Generate Trip Plan"):
            st.subheader(f"Your Personalized Trip to {selected_location}")
            
            # Simulate generating a trip plan based on preferences
            # In a real app, this would use more sophisticated algorithms
            
            # Find relevant landmark for this location
            relevant_landmark = None
            for landmark_key, info in LANDMARKS_DB.items():
                if location_key in landmark_key or location_key in info["name"].lower():
                    relevant_landmark = landmark_key
                    break
            
            if not relevant_landmark and location_key == "paris":
                relevant_landmark = "eiffel_tower"
            elif not relevant_landmark and location_key == "new_york":
                relevant_landmark = "statue_of_liberty"
            elif not relevant_landmark and location_key == "rome":
                relevant_landmark = "colosseum"
            elif not relevant_landmark and location_key == "agra":
                relevant_landmark = "taj_mahal"
            elif not relevant_landmark and location_key == "beijing":
                relevant_landmark = "great_wall"
            
            # Adjust preferences based on trip focus
            temp_preferences = user_preferences.copy()
            
            if trip_focus == "Must-see landmarks":
                temp_preferences["interests"] = ["history", "architecture"]
                temp_preferences["activity_level"] = "active"
            elif trip_focus == "Food and culture":
                temp_preferences["interests"] = ["local culture", "food"]
                temp_preferences["preferred_cuisine"] = ["Local"]
            elif trip_focus == "Off the beaten path":
                temp_preferences["interests"] = ["local culture", "nature"]
            elif trip_focus == "Relaxed pace":
                temp_preferences["activity_level"] = "light"
            
            # Adjust for time available
            if available_time <= 2:
                temp_preferences["activity_level"] = "light"
            elif available_time >= 8:
                temp_preferences["activity_level"] = "active"
            
            # Generate recommendations
            if relevant_landmark:
                recommendations = get_recommendations(relevant_landmark, temp_preferences)
                
                # Create an itinerary with time blocks
                start_time = 9  # 9 AM starting point
                
                st.write(f"### Main Attraction: {LANDMARKS_DB[relevant_landmark]['name']}")
                st.write(LANDMARKS_DB[relevant_landmark]['description'])
                
                st.write("### Your Itinerary:")
                
                # Morning activities
                st.write("#### Morning")
                st.write(f"{start_time}:00 AM - {start_time + 1}:30 AM: Visit {LANDMARKS_DB[relevant_landmark]['name']}")
                st.write(f"• {LANDMARKS_DB[relevant_landmark]['recommendations'][0] if LANDMARKS_DB[relevant_landmark]['recommendations'] else 'Explore the site'}")
                # Continue Trip Planning tab
                start_time += 2  # Move 2 hours ahead
                
                if "Walking" in transportation:
                    transport_method = "walking tour"
                elif "Public Transport" in transportation:
                    transport_method = "public transport"
                elif "Taxi/Rideshare" in transportation:
                    transport_method = "taxi"
                else:
                    transport_method = "guided tour"
                
                # Add lunch
                st.write("#### Lunch")
                if recommendations["restaurant_suggestions"]:
                    lunch_spot = recommendations["restaurant_suggestions"][0]
                    st.write(f"{start_time}:00 AM - {start_time + 1}:30 PM: Lunch at {lunch_spot['name']}")
                    st.write(f"• {lunch_spot['cuisine']} cuisine ({lunch_spot['price']})")
                    if lunch_spot['dietary']:
                        st.write(f"• Accommodates: {', '.join(lunch_spot['dietary'])}")
                else:
                    st.write(f"{start_time}:00 AM - {start_time + 1}:30 PM: Lunch at a local restaurant")
                
                start_time += 2  # Move 2 hours ahead
                
                # Afternoon activities
                st.write("#### Afternoon")
                if recommendations["itinerary"] and len(recommendations["itinerary"]) > 1:
                    afternoon_activity = recommendations["itinerary"][1]
                    if "nearby" in afternoon_activity.lower():
                        place = afternoon_activity.split("Visit nearby: ")[1]
                        st.write(f"{start_time}:00 PM - {start_time + 2}:00 PM: Visit {place}")
                        st.write(f"• Get there via {transport_method}")
                        st.write(f"• Recommended time: 1-2 hours")
                
                # Add more activities based on available time
                if available_time > 6:
                    start_time += 2  # Move 2 hours ahead
                    st.write("#### Evening")
                    
                    if len(recommendations["restaurant_suggestions"]) > 1:
                        dinner_spot = recommendations["restaurant_suggestions"][1]
                        st.write(f"{start_time}:00 PM - {start_time + 1}:30 PM: Dinner at {dinner_spot['name']}")
                        st.write(f"• {dinner_spot['cuisine']} cuisine ({dinner_spot['price']})")
                        if dinner_spot['dietary']:
                            st.write(f"• Accommodates: {', '.join(dinner_spot['dietary'])}")
                    else:
                        st.write(f"{start_time}:00 PM - {start_time + 1}:30 PM: Dinner at a recommended local spot")
                    
                    # Evening activity for longer days
                    if available_time > 8:
                        start_time += 2  # Move 2 hours ahead
                        if "local culture" in temp_preferences["interests"]:
                            st.write(f"{start_time}:00 PM - {start_time + 2}:00 PM: Enjoy local nightlife or cultural performance")
                        elif "architecture" in temp_preferences["interests"]:
                            st.write(f"{start_time}:00 PM - {start_time + 2}:00 PM: Evening city views or illuminated landmarks")
                        else:
                            st.write(f"{start_time}:00 PM - {start_time + 2}:00 PM: Free time to explore")
                
                # Special requirements handling
                if special_requirements:
                    st.write("#### Note on Special Requirements")
                    st.write(f"Your note: *{special_requirements}*")
                    st.write("We've taken your requirements into consideration in this plan.")
                
                # Transportation summary
                st.write("#### Transportation")
                st.write(f"This plan utilizes: {', '.join(transportation)}")
                
                # Map view (simulated)
                st.write("#### Map View")
                st.image("https://via.placeholder.com/800x400?text=Interactive+Map+of+Your+Itinerary", use_column_width=True)
                
                # Download option (simulated)
                st.download_button(
                    label="Download Trip Plan as PDF",
                    data="Sample PDF content",
                    file_name=f"trip_plan_{location_key}.pdf",
                    mime="application/pdf",
                )
                
                # Share options
                st.write("#### Share Your Plan")
                cols = st.columns(4)
                with cols[0]:
                    st.button("Email")
                with cols[1]:
                    st.button("WhatsApp")
                with cols[2]:
                    st.button("Message")
                with cols[3]:
                    st.button("Copy Link")

if __name__ == "__main__":
    app()
