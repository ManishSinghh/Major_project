import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

# Step 1: Load the pre-trained Random Forest model
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

# Step 2: Initialize the video capture
cap = cv2.VideoCapture(0)

# Step 3: Initialize the holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Step 4: Initialize the drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Step 5: Read the poses from the CSV file
df = pd.read_csv('Pose.csv')

# Step 6: Extract the pose features from the DataFrame
X = df.drop('class', axis=1)

# Step 7: Start the pose detection loop
while cap.isOpened():
    ret, frame = cap.read()
    
    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make Detections
    results = holistic.process(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract Pose landmarks
    pose_landmarks = results.pose_landmarks

    # Draw landmarks on the image
    if pose_landmarks:
        mp_drawing.draw_landmarks(image, pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Extract pose features
        pose_features = []
        for landmark in pose_landmarks.landmark:
            pose_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

        # Extract face features
        face_landmarks = results.face_landmarks
        if face_landmarks:
            face_features = []
            for landmark in face_landmarks.landmark:
                face_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            face_features = [0] * 468  # Set face features to zeros if not detected

        # Concatenate pose and face features
        row = pose_features + face_features

        # Make Detections
        X_new = pd.DataFrame([row])
        body_language_class = model.predict(X_new)[0]
        body_language_prob = model.predict_proba(X_new)[0]
        print(body_language_class, body_language_prob)

        # Display results on the image
        cv2.putText(image, f"Class: {body_language_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Probabilities: {body_language_prob}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Pose Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
