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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import threading
import json
import random
import subprocess
import sys
import streamlit as st
import requests
from openai import OpenAI

client=OpenAI(api_key="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")

import os
import subprocess
import sys
import streamlit as st


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

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.landmark_detected = None
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Simulated landmark detection
        # In a real app, this would use an actual CV model to detect landmarks
        # Here we're just randomly "detecting" landmarks occasionally
        if random.random() < 0.05:  # Small chance to detect a landmark each frame
            landmarks = list(LANDMARKS_DB.keys())
            self.landmark_detected = random.choice(landmarks)
            
            # Draw a box and label on the detected landmark
            height, width = img.shape[:2]
            cv2.rectangle(img, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), 2)
            cv2.putText(img, LANDMARKS_DB[self.landmark_detected]["name"], 
                      (width//4, height//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        return img

def process_image(image):
    """Process uploaded image for landmark detection"""
    # Simulated landmark detection for uploaded images
    # In a real app, this would use actual CV models
    landmarks = list(LANDMARKS_DB.keys())
    detected = random.choice(landmarks)
    
    # Convert the image to a format we can work with
    img_array = np.array(image)
    
    # Draw a box around the detected landmark
    height, width = img_array.shape[:2]
    cv2.rectangle(img_array, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), 2)
    cv2.putText(img_array, LANDMARKS_DB[detected]["name"], 
              (width//4, height//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
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
    """Capture and process voice input"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
        
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results"

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
                        st.image(processed_img, caption=f"Detected: {LANDMARKS_DB[detected_landmark]['name']}", use_column_width=True)
                        
                        # Show landmark information
                        st.subheader(LANDMARKS_DB[detected_landmark]['name'])
                        st.write(LANDMARKS_DB[detected_landmark]['description'])
                        
                        # Get personalized recommendations
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
        
        else:  # Use Camera
            st.write("Note: Camera access requires permission from your browser")
            
            ctx = webrtc_streamer(
                key="landmark-detection",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
            )
            
            if ctx.video_processor:
                if ctx.video_processor.landmark_detected:
                    detected_landmark = ctx.video_processor.landmark_detected
                    
                    if detected_landmark:
                        st.subheader(f"Detected: {LANDMARKS_DB[detected_landmark]['name']}")
                        st.write(LANDMARKS_DB[detected_landmark]['description'])
                        
                        # Get personalized recommendations
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

    def process_query_with_llm(query):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in LLM call: {e}")
            raise e  # Optional: re-raise if you want it to crash



    # Tab 2: Voice Assistant
    with tab2:
        st.header("Voice Assistant")
        st.write("Ask questions or get information using your voice (powered by an AI language model)")
        
        if st.button("Start Listening"):
            with st.spinner("Listening..."):
                try:
                    voice_text = listen_to_voice()
                    if voice_text and voice_text not in ["Could not understand audio", "Could not request results"]:
                        st.success(f"You said: {voice_text}")
                        
                        # Process the voice command with LLM
                        query = voice_text.lower()
                        response = process_query_with_llm(query)
                        st.write("Assistant:", response)
                        
                        # Post-process LLM response for specific actions
                        if "translate" in query.lower():
                            st.write("For translations, I’ll need a specific language. Please say something like 'Translate hello to French'.")
                        elif "map" in response.lower() or "location" in response.lower():
                            st.image("https://via.placeholder.com/600x400?text=Map+of+your+location", caption="Simulated Map")
                    
                    else:
                        st.error(voice_text or "No audio detected.")
                except Exception as e:
                    st.error(f"Error processing voice: {str(e)}")
        
        # Text input as fallback
        text_query = st.text_input("Or type your question:")
        if text_query:
            with st.spinner("Processing..."):
                response = process_query_with_llm(text_query)
                st.write("Assistant:", response)
                
                # Post-process LLM response for specific actions
                if "translate" in text_query.lower():
                    st.write("For translations, I’ll need a specific language. Please type something like 'Translate hello to French'.")
                elif "map" in response.lower() or "location" in response.lower():
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
            source_lang = "English"  # Fixed for simplicity
            target_lang = st.selectbox("Translate to:", ["French", "Spanish", "Italian", "Mandarin"])
        
        with col2:
            st.write("Translation:")
            if source_text and target_lang:
                translated_text = translate_text(source_text, target_lang.lower())
                st.text_area("Translation:", translated_text, height=150)
                
                # Text-to-speech option (simulated)
                if st.button("Listen to Translation"):
                    st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format="audio/mp3")
        
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
            
            translations = TRANSLATIONS.get(selected_phrase.lower(), {})
            
            with cols[0]:
                st.write("**French:**")
                st.write(translations.get("french", "-"))
            
            with cols[1]:
                st.write("**Spanish:**")
                st.write(translations.get("spanish", "-"))
            
            with cols[2]:
                st.write("**Italian:**")
                st.write(translations.get("italian", "-"))
            
            with cols[3]:
                st.write("**Mandarin:**")
                st.write(translations.get("mandarin", "-"))

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
