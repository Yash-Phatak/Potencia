import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras import models
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pathlib
import textwrap
import google.generativeai as genai 
from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

model = keras.models.load_model('gym_classification_model.h5')

class_indices = {'Dumbells': 0, 'Elliptical': 1, 'Home Machine': 2, 'Recumbent Bike': 3}
class_labels = {v: k for k, v in class_indices.items()}

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

img_path = 'gym-data/test_image_5.jpg'

predicted_class, confidence = classify_image(model, img_path)
print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")



GOOGLE_API_KEY = 'AIzaSyAcIimOXgOOEMMIcSyUhZ_RoOtKSe38VRY'
genai.configure(api_key=GOOGLE_API_KEY)

gemini = genai.GenerativeModel('gemini-1.5-flash')
response = gemini.generate_content("Give a workout plan for the detected equipment: "+ predicted_class)

try:
    response = gemini.generate_content("Give a workout plan for the detected equipment: " + predicted_class)
    print(f"Response: {response}")
    
    if hasattr(response, 'text'):
        workout_plan = response.result.candidates[0].content.parts[0].text
        print(workout_plan)
    else:
        print("No text in the response.")
except Exception as e:
    print(f"An error occurred: {e}")
