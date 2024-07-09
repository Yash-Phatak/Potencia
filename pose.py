import cv2
import mediapipe as mp
import numpy as np
import math
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a,b,c):
    a = np.array(a) #First point
    b = np.array(b) #Mid point about which angle is found
    c = np.array(c) #End Point 
     
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle = 360-angle
    return angle
def perp_distance(a,b,c):
    num = abs((a[0]-b[0])*(c[1]-b[1])-(a[1]-b[1])*(c[0]-b[0]))
    denom = math.sqrt(math.pow(a[0]-b[0],2) + math.pow(a[1]-b[1],2))
    return num/denom

# Right and Left Biceps
def biceps():
    cap = cv2.VideoCapture(0)

    # Curl Counter Variables
    left_counter = 0
    right_counter = 0
    left_stage = None
    right_stage = None
    # Setting up Mediapipe Instance 
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret,frame = cap.read() #Start Capturing
            # Recolor to RGB
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False   

            # Detecting and storing in results list
            results = pose.process(image)

            # Recolor back to BGR
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            
            try:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Get the required coordinates  
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Get the Angle
                left_angle = calculate_angle(left_shoulder,left_elbow,left_wrist)
                right_angle = calculate_angle(right_shoulder,right_elbow,right_wrist)
                # Visualise 
                # Left
                cv2.putText(image,str(left_angle),
                            tuple(np.multiply(left_elbow,[640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                # Right
                cv2.putText(image,str(right_angle),tuple(np.multiply(right_elbow,[640.480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                # Landmark Rendering
                mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(117,66,245), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
                
                # Curl Counter  
                if left_angle > 150:
                    left_stage = "down"
                if left_angle < 30 and left_stage == "down":
                    left_stage="up"
                    left_counter +=1
                    print("Left Counter: {}".format(left_counter))
                if right_angle > 150:
                    right_stage = "down"
                if right_angle < 30 and right_stage=="down":
                    right_stage = "up"
                    right_counter+=1
                    print("Right Counter: {}".format(right_counter))

            except: pass
            image = cv2.flip(image, 1) # Subject to Approval
            # Rendering Curl Counter 
            # Setting up Status Box
            # Set a dark color for the text
            dark_text_color = (10, 10, 10)
            # dark_text_color = (255,255,255)
            # For Left: 
            # Rep data
            cv2.putText(image, 'LEFT REPS', (15, 30), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_counter), 
                        (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, dark_text_color, 2, cv2.LINE_AA)    
            # Stage data
            cv2.putText(image, 'STAGE', (15, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, left_stage, 
                        (50, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, dark_text_color, 2, cv2.LINE_AA)
            
            # For Right:
            cv2.putText(image, 'RIGHT REPS', (465, 30), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_counter), 
                        (460, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, dark_text_color, 2, cv2.LINE_AA)    
            # Stage data
            cv2.putText(image, 'STAGE', (465, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, right_stage, 
                        (500, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, dark_text_color, 2, cv2.LINE_AA)
            cv2.imshow('Mediapipe Feed',image)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# Jump Counter
def jump():
    cap = cv2.VideoCapture(0)
    p1 = []
    p2 = []
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret,frame = cap.read()
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
                break

            except:
                print("No")
                pass
        while cap.isOpened():
            ret,frame = cap.read()
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
            # cv2.putText(image, f"({int(p1[0])}, {int(p1[1])})", (int(p1[0]), int(p1[1])),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, f"({int(p2[0])}, {int(p2[1])})", (int(p2[0]), int(p2[1])),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
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
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# Neck Movements
def neck():
    cap = cv2.VideoCapture(0)
    stage = ""
    with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret,frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark

                #Getting coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                angle = calculate_angle(left_shoulder,right_shoulder,nose)
                cv2.putText(image,str(angle),tuple(np.multiply(right_shoulder,[640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(117,66,245), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
                
                if angle > 65:
                    stage = "Right"
                if angle < 45:
                    stage = "Left"
            except: pass
            dark_text_color = (10,10,10)
            cv2.putText(image, 'NECK: ', (65, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, dark_text_color, 2, cv2.LINE_AA)
            cv2.imshow('Mediapipe Feed',image)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

biceps()










def potencia(frame):
    # Curl Counter Variables
    global left_counter, right_counter, left_stage, right_stage

    if 'left_counter' not in globals():
        left_counter = 0
    if 'right_counter' not in globals():
        right_counter = 0
    if 'left_stage' not in globals():
        left_stage = None
    if 'right_stage' not in globals():
        right_stage = None

    # Setting up Mediapipe Instance 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Recolor to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detecting and storing in results list
        results = pose.process(image)

        # Recolor back to BGR
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Get the required coordinates  
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Get the Angle
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Visualise 
            # Left
            # cv2.putText(image, str(left_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            # # Right
            # cv2.putText(image, str(right_angle), tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # # Landmark Rendering
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #                           mp_drawing.DrawingSpec(color=(117, 66, 245), thickness=2, circle_radius=2),
            #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2))
            
            # Curl Counter  
            if left_angle > 150:
                left_stage = "down"
            if left_angle < 30 and left_stage == "down":
                left_stage = "up"
                left_counter += 1
                print("Left Counter: {}".format(left_counter))
            if right_angle > 150:
                right_stage = "down"
            if right_angle < 30 and right_stage == "down":
                right_stage = "up"
                right_counter += 1
                print("Right Counter: {}".format(right_counter))

        except Exception as e:
            print(f"Error processing frame: {e}")

        # Display frame with results
        # image = cv2.flip(image, 1)  # Flip horizontally for a mirror effect
        # cv2.imshow('Mediapipe Feed', image)
        return left_counter,right_counter