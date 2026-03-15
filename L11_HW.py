import cv2
import numpy as np
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
model = load_model('emotion_model.hdf5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: CANNOT OPEN CAMERA")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: FAILED TO CAPTURE IMAGE")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1,48,48,1))
        prediction = model.predict(roi_gray, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)
    cv2.imshow('Face & Emotion Detection - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()