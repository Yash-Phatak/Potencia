import mediapipe as mp
import cv2
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pickle
import time
import pandas as pd
import math
import warnings
warnings.filterwarnings('ignore')
mp_drawing  = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']  # Class is Y - target value
for val in range(1,33+1): # 32 total landmarks
    landmarks+=['x{}'.format(val),'y{}'.format(val),'z{}'.format(val),'v{}'.format(val)]

with open('deadlift\pickle files\deadlift.pkl','rb') as f:
    deadlift_model = pickle.load(f)
with open('deadlift\pickle files\lean.pkl','rb') as f:
    lean_model = pickle.load(f)
with open('deadlift\pickle files\distance.pkl','rb') as f:
    distance_model = pickle.load(f)

def calculate_distance(a,b):
    return np.sqrt((b[0] - a[0])**2 + (b[1]-a[1])**2)

def perp_distance(a,b,c):
    num = abs((a[0]-b[0])*(c[1]-b[1])-(a[1]-b[1])*(c[0]-b[0]))
    denom = math.sqrt(math.pow(a[0]-b[0],2) + math.pow(a[1]-b[1],2))
    return num/denom

# Constants for thresholds
LEAN_CONFIDENCE_THRESHOLD = 0.7
BODYLANG_CONFIDENCE_THRESHOLD_DOWN = 0.7
BODYLANG_CONFIDENCE_THRESHOLD_UP = 0.7
DISTANCE_CONFIDENCE_THRESHOLD = 0.7

selected_landmarks = [] #  note in all the databases the landmarks are starting from 1 instead of 0
for i in range(23, 33):
    selected_landmarks += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

def deadlift(frame):
    cap = cv2.VideoCapture(0)
    current_Stage = ''
    current_lean_Status = ''
    current_distance_Status = ''
    counter = 0 
    time.sleep(3)
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        # Recolor Feed
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = pose.process(image)
        # Recolor it back
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS
                                , mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))
        
        try:
            # Distance things
            landmarks_mp = results.pose_landmarks.landmark
            left_knee = [landmarks_mp[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks_mp[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks_mp[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks_mp[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            distance = calculate_distance(left_knee,right_knee)*100

            row = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row],columns=landmarks[1:])
            row_dist = row[(23*4):]
            X_dist = pd.DataFrame([row_dist],columns=selected_landmarks)
            bodylang_prob = deadlift_model.predict_proba(X)[0]
            bodylang_class = deadlift_model.predict(X)[0]
            lean_prob = lean_model.predict_proba(X)[0]
            lean_class = lean_model.predict(X)[0]
            distance_prob = distance_model.predict_proba(X_dist)[0]
            distance_class = distance_model.predict(X_dist)[0]
            # Check if the posture is centered with sufficient confidence
            if (lean_class == 1.0 and lean_prob[lean_prob.argmax()] > LEAN_CONFIDENCE_THRESHOLD) or (flag==0) and distance>12 and distance<18:
                flag=0
                # Update the stage based on body language class and confidence
                if bodylang_class == 0.0 and bodylang_prob[bodylang_prob.argmax()] > BODYLANG_CONFIDENCE_THRESHOLD_DOWN:
                    current_Stage = "down"
                elif bodylang_class == 1.0 and current_Stage == "down" and bodylang_prob[bodylang_prob.argmax()] > BODYLANG_CONFIDENCE_THRESHOLD_UP:
                    current_Stage = "up"
                    counter += 1

            # Update distance status
            if distance<11:
                current_distance_Status = "Narrow"
            elif distance>=11 and distance<18:
                current_distance_Status = "Correct"
            else: distance = "Wide"

            # Update lean status
            if lean_class == 0.0 and lean_prob[lean_prob.argmax()] > LEAN_CONFIDENCE_THRESHOLD and current_Stage=="up":
                current_lean_Status = "Left"
            elif lean_class == 1.0 and lean_prob[lean_prob.argmax()] > LEAN_CONFIDENCE_THRESHOLD:
                current_lean_Status = "Centre"
            elif lean_class == 2.0 and lean_prob[lean_prob.argmax()] > LEAN_CONFIDENCE_THRESHOLD and current_Stage=="down":
                current_lean_Status = "Right"
            else: current_lean_Status = "Centre"
            
            # Get status box
            cv2.rectangle(image,(0,0),(550,60),(245,117,16),-1)

            # Display Rep
            cv2.putText(image,'Status',
                        (95,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,current_Stage,
                            (90,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            # Display Probability
            cv2.putText(image,'Prob',
                        (15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(round(bodylang_prob[np.argmax(bodylang_prob)],2)),
                        (10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            # Display Count
            cv2.putText(image,'Count',
                        (180,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(counter),
                        (200,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            # Display Lean
            cv2.putText(image,'Posture',
                        (280,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,current_lean_Status,
                            (280,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            # Display Distance
            cv2.putText(image,'Distance',
                        (400,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,current_distance_Status,
                            (400,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('Deadlifts',image)
        except Exception as e:
            print(e)
    return current_lean_Status,current_distance_Status,current_Stage,counter
            
            

def jump(frame):
    p1 = []
    p2 = []
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image.flags.writeable = True
        try:
            landmarks = results.pose_landmarks.landmark
            a1 = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value] 
            a2 = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
            p1 = [a1.x*frame.shape[1],a1.y*frame.shape[0]]
            p2 = [a2.x*frame.shape[1],a2.y*frame.shape[0]]  
            print(p1[0])

        except:
            print("No")
            pass
    time.sleep(3)
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        #Recolour the image to RGB to make it compatible with Mediapipe
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Make Detection
        results = pose.process(image)

        #Recolor back to BGR to make it compatible with Open CV
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            median_x = int((left_eye[0]+right_eye[0])*frame.shape[1]/2)
            median_y = int((left_eye[1]+right_eye[1])*frame.shape[0]/2)
            median = [median_x,median_y]
            dist = perp_distance(median,p1,p2)
            if dist<100:
                stage = "JUMP"
            if dist>300 and stage=="JUMP":
                counter+=1
                stage = ""
            cv2.putText(image, str(dist), (465, 30), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, f"({median_x}, {median_y})", (median_x, median_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            print("Error")
            pass
        #Render Detections
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        # Counters
        cv2.putText(image,'Counter',(15,30),
                    cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(10,120),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),1,cv2.LINE_AA)
        # Stage
        cv2.putText(image,'STAGE',(15,65),
                    cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image,stage,(50,120),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),1,cv2.LINE_AA)
        # Reference Line 
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
        cv2.imshow('Mediapipe Feed',image)
        return stage, counter
