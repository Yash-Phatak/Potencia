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
    recommendations = []
    tfidf = TfidfVectorizer(max_features=2000)
    X = tfidf.fit_transform(data['string'])
    user_tfidf = tfidf.transform([text])
    scores = cosine_similarity(user_tfidf,X).flatten()
    recommend_idx = (-scores).argsort()[:10]
    for idx in recommend_idx:
        name = data['Title'].iloc[idx]
        desc = data['Desc'].iloc[idx]
        target_muscle = data['BodyPart'].iloc[idx]
        level = data['Level'].iloc[idx]
        type = data['Type'].iloc[idx]
        equipment = data['Equipment'].iloc[idx]
        recommendations.append({"name":name,"desc":desc,"target_muscle":target_muscle,"level":level,"type":type,"equipment":equipment})
    return recommendations

def process(input_dict):
    output_list = []
    tm = list(input_dict.values())[1]
    for i in range(len(tm)):
        input_string = tm[i] + " " + input_dict["level"] + " " +' '.join(list(input_dict.values())[3])
        output = recommend(input_string)
        output_list.append(output)
    l = len(output_list)
    keys = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    output_dict = {"uid":input_dict["uid"]}
    for i,key in enumerate(keys):
        output_dict[key] = output_list[i%l]
    return output_dict
if __name__=='__main__':
    exercise = input("String from Frontend: ")
    print(process())