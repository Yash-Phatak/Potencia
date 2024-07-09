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
from prompt import prompt_text
from PIL import Image
import json
import re

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

model = keras.models.load_model('equip_detection\gym_classification_model.h5')

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

def detect(image: Image.Image) -> dict:
    img_path = 'imagesentfromfrontend.jpg'
    image.save(img_path)

    predicted_class, confidence = classify_image(model,img_path)
    print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")



    GOOGLE_API_KEY = 'AIzaSyAcIimOXgOOEMMIcSyUhZ_RoOtKSe38VRY'
    genai.configure(api_key=GOOGLE_API_KEY)

    gemini = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Give a workout plan  (name,description, warm up, reps , target muscles ) for the detected equipment in json documented format: {predicted_class}. Strictly maintain only one  json format. Every field in json is in string datatype."
    response = gemini.generate_content(prompt)
    # response = response.to_dict()
    # print(response)
    response = response.to_dict()
    text = response['candidates'][0]['content']['parts'][0]['text']

    with open('output.txt', 'w') as f:
        f.write(text)

    # # Reading the response
    # with open('output.txt', 'r') as file:
    #     content = file.read()

    # pattern = re.compile(r'({.*?})', re.DOTALL)
    # match = pattern.search(content)

    # if match:
    #     json_string = match.group(1)
    #     try:
    #         json_data = json.loads(json_string)
    #         with open('output.json', 'w') as json_file:
    #             json.dump(json_data, json_file, indent=4)
            
    #         print("JSON data successfully extracted and saved to 'output.json'.")
    #     except json.JSONDecodeError as e:
    #         print("Invalid JSON data:", e)
    # else:
    #     print("No JSON content found")

    return response

def detect(image:Image.Image) ->dict:
    img_path = 'imagesentfromfrontend.jpg'
    image.save(img_path)

    predicted_class, confidence = classify_image(model,img_path)
    print(f"Predicted class: {predicted_class} with confidence {confidence:.2f}")



    GOOGLE_API_KEY = 'AIzaSyAcIimOXgOOEMMIcSyUhZ_RoOtKSe38VRY'
    genai.configure(api_key=GOOGLE_API_KEY)

    gemini = genai.GenerativeModel('gemini-1.5-flash')
    prompt = prompt_text(predicted_class)
    response = gemini.generate_content(prompt)
    print(response)
    response = response.to_dict()
    text = response['candidates'][0]['content']['parts'][0]['text']

    with open('output.txt', 'w') as f:
        f.write(text)

    with open('output.txt', 'r') as file:
        lines = file.readlines()
    
    text = ''.join(lines[1:-1])

    plan = json.loads(text)

    with open('plan.json','w') as f:
        json.dump(plan,f,indent=4)

    return response

detect()