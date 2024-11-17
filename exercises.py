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
LEAN_CONFIDENCE_THRESHOLD = 0.3
BODYLANG_CONFIDENCE_THRESHOLD_DOWN = 0.5
BODYLANG_CONFIDENCE_THRESHOLD_UP = 0.7
DISTANCE_CONFIDENCE_THRESHOLD = 0.7

selected_landmarks = [] #  note in all the databases the landmarks are starting from 1 instead of 0
for i in range(23, 33):
    selected_landmarks += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

current_Stage = "up"
current_lean_Status = ''
current_distance_Status = ''
counter = 0

def deadlift(frame):
    global current_Stage, current_lean_Status, current_distance_Status, counter
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        # Recolor Feed
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = pose.process(image)
        # Recolor it back
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        try:
            # Distance things
            landmarks_mp = results.pose_landmarks.landmark
            left_knee = [landmarks_mp[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks_mp[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks_mp[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks_mp[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            distance = calculate_distance(left_knee,right_knee)*100

            row = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row],columns=landmarks[1:])
            row_dist = row[(23*4):]
            # X_dist = pd.DataFrame([row_dist],columns=selected_landmarks)
            bodylang_prob = deadlift_model.predict_proba(X)[0]
            bodylang_class = deadlift_model.predict(X)[0]
            lean_prob = lean_model.predict_proba(X)[0]
            lean_class = lean_model.predict(X)[0]
            # Check if the posture is centered with sufficient confidence
            if ((lean_class == 1.0 or lean_class==0.0) and lean_prob[lean_prob.argmax()] > LEAN_CONFIDENCE_THRESHOLD) and distance>12 and distance<18:
                # Update the stage based on body language class and confidence
                if bodylang_class == 0.0 and bodylang_prob[bodylang_prob.argmax()] > BODYLANG_CONFIDENCE_THRESHOLD_DOWN:
                    current_Stage = "down"
                elif bodylang_class == 1.0 and bodylang_prob[bodylang_prob.argmax()] > BODYLANG_CONFIDENCE_THRESHOLD_UP:
                    if current_Stage == "down":
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
                current_lean_Status = "Centre"
            elif lean_class == 1.0 and lean_prob[lean_prob.argmax()] > LEAN_CONFIDENCE_THRESHOLD:
                current_lean_Status = "Centre"
            elif lean_class == 2.0 and lean_prob[lean_prob.argmax()] > LEAN_CONFIDENCE_THRESHOLD and current_Stage=="up":
                current_lean_Status = "Right"
            else: current_lean_Status = "Centre"
        except Exception as e:
            print(e)
    return current_lean_Status,current_distance_Status,current_Stage,counter
            
# flag= 0 
# p1 = []
# p2 = []        

# def jump(frame):
#     global p1,p2
#     global counter
#     global flag
#     counter = 0
#     global stage 
#     stage = 0
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = pose.process(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         image.flags.writeable = True

#         if(flag<5):
#             try:
#                 landmarks = results.pose_landmarks.landmark
#                 left_toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
#                 right_toe = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
#                 p1 = [left_toe.x * frame.shape[1], left_toe.y * frame.shape[0]]
#                 p2 = [right_toe.x * frame.shape[1], right_toe.y * frame.shape[0]]
#                 print("Reference line drawn between:", p1, p2)
#                 flag+=1
#             except:
#                 print("No pose landmarks detected.")
#                 pass
#         else:
#             try:
#                 landmarks = results.pose_landmarks.landmark
#                 left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#                 right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#                 hip_median_x = int((left_hip[0] + right_hip[0]) * frame.shape[1] / 2)
#                 hip_median_y = int((left_hip[1] + right_hip[1]) * frame.shape[0] / 2)
#                 hip_median = [hip_median_x, hip_median_y]
#                 dist = perp_distance(hip_median, p1, p2)
#                 print(dist)
#                 if dist < 80:
#                     stage = "JUMP"
#                 elif dist > 100 and stage == "JUMP":
#                     counter += 1
#                     stage = "AIR"

#                 # cv2.putText(image, str(dist), (465, 30), 
#                 #             cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
#                 # cv2.putText(image, f"({hip_median_x}, {hip_median_y})", (hip_median_x, hip_median_y),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#             except:
#                 print("Error in processing pose landmarks.")
#                 pass

#             # Render Detections
#             # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#             #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

#             # Display counter and stage on the image (optional)
#             # cv2.putText(image, 'Counter', (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
#             # cv2.putText(image, str(counter), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1, cv2.LINE_AA)
#             # cv2.putText(image, 'STAGE', (15, 65), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
#             # cv2.putText(image, stage, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1, cv2.LINE_AA)

#             # Display reference line
#             # if p1 and p2:
#             #     cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

#             # # Display the image
#             # cv2.imshow('Mediapipe Feed', image)
#     return stage,counter



def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialize global variables
counter = 0
stage = None

def jump(frame):
    global counter, stage

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True
        landmarks = results.pose_landmarks.landmark
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_angle = calculate_angle(left_shoulder, left_hip, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_hip, right_wrist)
        print(left_angle,right_angle)
        if left_angle > 65 and right_angle > 65:
            stage = "down"
        if left_angle < 45 and right_angle < 45 and stage == "down":
            stage = "up"
            counter += 1


    return stage, counter