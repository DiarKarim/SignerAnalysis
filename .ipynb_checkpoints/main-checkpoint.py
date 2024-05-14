import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import time

# May 14th 2024
import pandas as pd
import seaborn as sns


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_holistic.POSE_CONNECTIONS
mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
mp_drawing.draw_landmarks

# Setup variabls
face_data_x = []
face_data_y = []
face_data_z = []

hand_left_x = []
hand_left_y = []
hand_left_z = []

# hand_right = []

left_hand_x, left_hand_y, left_hand_z = 0,0,0

# face_time = []
# hand_left_time = []

time_capture = []

tmp1_data_x, tmp2_data_x, tmp3_data_x, tmp4_data_x, tmp5_data_x, tmp6_data_x, tmp7_data_x, tmp8_data_x = 0,0,0,0,0,0,0,0
tmp1_data_y, tmp2_data_y, tmp3_data_y, tmp4_data_y, tmp5_data_y, tmp6_data_y, tmp7_data_y, tmp8_data_y = 0,0,0,0,0,0,0,0
tmp1_data_z, tmp2_data_z, tmp3_data_z, tmp4_data_z, tmp5_data_z, tmp6_data_z, tmp7_data_z, tmp8_data_z = 0,0,0,0,0,0,0,0

rbrow_x, rbrow_y, rbrow_z = [],[],[]
lbrow_x, lbrow_y, lbrow_z = [],[],[]
tophead_x, tophead_y, tophead_z = [],[],[]
leftcheeck_x, leftcheeck_y, leftcheeck_z = [],[],[]
right_cheeck_x, right_cheeck_y, right_cheeck_z = [],[],[]
chin_x, chin_y, chin_z = [],[],[]
uppmidlip_x, uppmidlip_y, uppmidlip_z = [],[],[]
bottmidlip_x, bottmidlip_y, bottmidlip_z = [],[],[]

# Create a loop to go through the videos 
path = "D:/Project/SignerAnalysis/rawvideos/"
file = "example.mp4" 

cap = cv2.VideoCapture(path + file)
# cap = cv2.VideoCapture(1)

fps = cap.get(cv2.CAP_PROP_FPS) 
framenumber = 0

print("FPS: " + str(fps))

startTime = time.time()

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        try:
            # ret, frame = cap.read()
            
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
                
            
            framenumber = framenumber + 1

            # Recolor Feed
            image_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make Detections
            results = holistic.process(image_mp)
            # print(results.face_landmarks)

            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image_mp, cv2.COLOR_RGB2BGR)

            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )

            # 5. Save face data        
            # Extract specific landmarks 
            if results.face_landmarks:
                for landmark in results.face_landmarks.landmark: 
                    rightBrowIndex = 105
                    tmp1_data_x = results.face_landmarks.landmark[rightBrowIndex].x
                    tmp1_data_y = results.face_landmarks.landmark[rightBrowIndex].y
                    tmp1_data_z = results.face_landmarks.landmark[rightBrowIndex].z

                    leftBrowIndex = 334
                    tmp2_data_x = results.face_landmarks.landmark[leftBrowIndex].x
                    tmp2_data_y = results.face_landmarks.landmark[leftBrowIndex].y
                    tmp2_data_z = results.face_landmarks.landmark[leftBrowIndex].z

                    tophead = 10
                    tmp3_data_x = results.face_landmarks.landmark[tophead].x
                    tmp3_data_y = results.face_landmarks.landmark[tophead].y
                    tmp3_data_z = results.face_landmarks.landmark[tophead].z

                    left_cheeck = 454
                    tmp4_data_x = results.face_landmarks.landmark[left_cheeck].x
                    tmp4_data_y = results.face_landmarks.landmark[left_cheeck].y
                    tmp4_data_z = results.face_landmarks.landmark[left_cheeck].z

                    right_cheeck = 234
                    tmp5_data_x = results.face_landmarks.landmark[right_cheeck].x
                    tmp5_data_y = results.face_landmarks.landmark[right_cheeck].y
                    tmp5_data_z = results.face_landmarks.landmark[right_cheeck].z

                    chin = 152
                    tmp6_data_x = results.face_landmarks.landmark[chin].x
                    tmp6_data_y = results.face_landmarks.landmark[chin].y
                    tmp6_data_z = results.face_landmarks.landmark[chin].z

                    uppermiddlelip = 0
                    tmp7_data_x = results.face_landmarks.landmark[uppermiddlelip].x
                    tmp7_data_y = results.face_landmarks.landmark[uppermiddlelip].y
                    tmp7_data_z = results.face_landmarks.landmark[uppermiddlelip].z

                    bottommiddlelip = 17
                    tmp8_data_x = results.face_landmarks.landmark[bottommiddlelip].x
                    tmp8_data_y = results.face_landmarks.landmark[bottommiddlelip].y
                    tmp8_data_z = results.face_landmarks.landmark[bottommiddlelip].z


            rbrow_x.append(tmp1_data_x)
            rbrow_y.append(tmp1_data_y)
            rbrow_z.append(tmp1_data_z)

            lbrow_x.append(tmp2_data_x)
            lbrow_y.append(tmp2_data_y)
            lbrow_z.append(tmp2_data_z)

            tophead_x.append(tmp3_data_x)
            tophead_y.append(tmp3_data_y)
            tophead_z.append(tmp3_data_z)

            leftcheeck_x.append(tmp4_data_x)
            leftcheeck_y.append(tmp4_data_y)
            leftcheeck_z.append(tmp4_data_z)

            right_cheeck_x.append(tmp5_data_x)
            right_cheeck_y.append(tmp5_data_y)
            right_cheeck_z.append(tmp5_data_z)

            chin_x.append(tmp6_data_x)
            chin_y.append(tmp6_data_y)
            chin_z.append(tmp6_data_z)

            uppmidlip_x.append(tmp7_data_x)
            uppmidlip_y.append(tmp7_data_y)
            uppmidlip_z.append(tmp7_data_z)

            bottmidlip_x.append(tmp8_data_x)
            bottmidlip_y.append(tmp8_data_y)
            bottmidlip_z.append(tmp8_data_z)

            
            duration = (framenumber/fps) % 60             
            time_capture.append(duration) 


            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                     )

            # 5. Save face data        
            # Extract specific landmarks 
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark: 
                    left_hand_x = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x
                    left_hand_y = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y
                    left_hand_z = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z

            hand_left_x.append(left_hand_x)
            hand_left_y.append(left_hand_y)
            hand_left_z.append(left_hand_z)


            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )

            
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(e)

            
data_dictionary = {"rbrow_x" : rbrow_x,
                   "rbrow_y" : rbrow_y, 
                   "rbrow_z" : rbrow_z, 
                   "lbrow_x" : lbrow_x,
                   "lbrow_y" : lbrow_y, 
                   "lbrow_z" : lbrow_z, 
                   "tophead_x" : tophead_x,
                   "tophead_y" : tophead_y, 
                   "tophead_z" : tophead_z,   
                   "leftcheeck_x" : leftcheeck_x,
                   "leftcheeck_y" : leftcheeck_y, 
                   "leftcheeck_z" : leftcheeck_z,     
                   "right_cheeck_x" : right_cheeck_x,
                   "right_cheeck_y" : right_cheeck_y, 
                   "right_cheeck_z" : right_cheeck_z,   
                   "chin_x" : chin_x,
                   "chin_y" : chin_y, 
                   "chin_z" : chin_z,  
                   "uppmidlip_x" : uppmidlip_x,
                   "uppmidlip_y" : uppmidlip_y, 
                   "uppmidlip_z" : uppmidlip_z,                     
                   "bottmidlip_x" : bottmidlip_x,
                   "bottmidlip_y" : bottmidlip_y, 
                   "bottmidlip_z" : bottmidlip_z,     
                   "lhand_x" : hand_left_x,
                   "lhand_y" : hand_left_y, 
                   "lhand_z" : hand_left_z, 
                   "time" : time_capture,}

df = pd.DataFrame(data_dictionary)
df['videotrial'] = 0 
df['condition'] = 1 
df['ptxid'] = 0 


df.to_pickle('singerdata.pkl')


# plt.figure()
# sns.lineplot(x='time', y = 'lhand_y', data=df)
# plt.show()

