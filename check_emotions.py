'''
PyPower Projects
Emotion Detection Using AI
'''

# USAGE: python check_emotions.py

from keras.models import load_model
from time import time
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load Haar Cascade and Emotion Detection Model
face_classifier = cv2.CascadeClassifier('AI Smart_face_detector.xml')
classifier = load_model('SmartEmotionModel_v1.h5')

# Emotion labels
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion-specific colors
emotion_colors = {
    'Angry': (0, 0, 255),
    'Happy': (0, 255, 255),
    'Neutral': (200, 200, 200),
    'Sad': (128, 0, 128),
    'Surprise': (0, 255, 0)
}

# Start webcam
cap = cv2.VideoCapture(0)
prev_time = time()

while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Yellow rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion and get confidence
            preds = classifier.predict(roi)[0]
            emotion_index = preds.argmax()
            label = class_labels[emotion_index]
            confidence = int(preds[emotion_index] * 100)
            label_text = f'{label} - {confidence}%'
            label_position = (x, y - 10)
            color = emotion_colors[label]

            # Show label and accuracy in red
            cv2.putText(frame, label_text, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Graph bar (under face)
            bar_x1, bar_y1 = x, y + h + 10
            bar_x2 = x + int((w) * (confidence / 100))
            bar_y2 = y + h + 30
            cv2.rectangle(frame, (bar_x1, bar_y1), (x + w, bar_y2), (50, 50, 50), -1)  # background
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), color, -1)        # confidence

        else:
            cv2.putText(frame, 'No Face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Show FPS in green
    curr_time = time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('AI Smart MoodVision', frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
