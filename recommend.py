import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = pd.read_csv('data/gym_data.csv')

def to_String(row):
    title = row['Title']
    type = row['Type']
    desc = row['Desc']
    bodypart = row['BodyPart']
    equipment = row['Equipment']
    level = row['Level']
    return "%s %s %s %s %s %s"%(title,desc,type,bodypart,equipment,level)
data['string'] = data.apply(to_String,axis=1)
# Final Function
def recommend(text):
    tfidf = TfidfVectorizer(max_features=2000)
    X = tfidf.fit_transform(data['string'])
    user_tfidf = tfidf.transform([text])
    scores = cosine_similarity(user_tfidf,X).flatten()
    recommend_idx = (-scores).argsort()[1:11]
    recommendation = list(data['Title'].iloc[recommend_idx])
    output = {"answer":recommendation}
    return output


if __name__=='__main__':
    exercise = input("String from Frontend: ")
    print(recommend(exercise))