import cv2
import mediapipe as mp
import numpy as np
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

def biceps():
    cap = cv2.VideoCapture(0)

    # Curl Counter Variables
    counter = 0
    stage = None
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
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Get the Angle
                angle = calculate_angle(shoulder,elbow,wrist)

                # Visualise 
                cv2.putText(image,str(angle),
                            tuple(np.multiply(elbow,[640,480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
                # Landmark Rendering
                mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(117,66,245), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
                
                # Curl Counter
                if angle > 150:
                    stage = "down"
                if angle < 30 and stage == "down":
                    stage="up"
                    counter +=1
                    print(counter)

            except: pass

            # Rendering Curl Counter 
            #Setting up Status Box
            # Set a dark color for the text
            dark_text_color = (10, 10, 10)

            # Rep data
            cv2.putText(image, 'REPS', (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, dark_text_color, 2, cv2.LINE_AA)
                    
            # Stage data
            cv2.putText(image, 'STAGE', (65, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, dark_text_color, 2, cv2.LINE_AA)
            
            # cv2.imshow('Mediapipe Feed',image)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        # cap.release()
        # cv2.destroyAllWindows()

biceps()