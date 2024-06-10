import pandas as pd
import numpy as np
import json
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

# Load the data
data = pd.read_csv('data/gym_data.csv')

# Convert row to a string for TF-IDF processing
def to_string(row):
    return f"{row['Title']} {row['Desc']} {row['Type']} {row['BodyPart']} {row['Equipment']} {row['Level']}"

data['string'] = data.apply(to_string, axis=1)
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(data['string'])

# Recommendation
def recommend(text, data):
    user_tfidf = tfidf.transform([text])
    scores = cosine_similarity(user_tfidf, X).flatten()
    recommend_idx = (-scores).argsort()[:]
    
    recommendations = []
    for idx in recommend_idx:
        recommendations.append({
            "name": data['Title'].iloc[idx],
            "desc": data['Desc'].iloc[idx],
            "target_muscle": data['BodyPart'].iloc[idx],
            "level": data['Level'].iloc[idx],
            "type": data['Type'].iloc[idx],
            "equipment": data['Equipment'].iloc[idx]
        })
    
    return recommendations


# Plan Processing 
def process(input_dict):
    target_muscles = input_dict['target_muscle']
    level = input_dict['level']
    types = ' '.join(input_dict['type'])

    output_dict = {"uid": input_dict['uid']}
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    
    all_recommendations = []
    for muscle in target_muscles:
        input_string = f"{muscle} {level} {types}"
        recommendations = recommend(input_string, data)
        all_recommendations.extend(recommendations)
    daily_recommendations = [all_recommendations[i:(i+5)] for i in range(0, len(all_recommendations), 5)]

    # 5 recommendations
    for i, day in enumerate(days):
        output_dict[day] = daily_recommendations[i % len(daily_recommendations)]
    
    return output_dict

# Sample input dictionary
class RequestBody(BaseModel):
    uid: str
    target_muscle: List[str]
    level: str
    type: List[str]

input_dict = RequestBody(
    uid="user123",
    target_muscle=["Chest", "Back", "Legs"],
    level="Intermediate",
    type=["Dumbbells", "Barbell", "Bench"]
)

if __name__ == '__main__':
    output = process(input_dict)
    print(json.dumps(output, indent=4))
