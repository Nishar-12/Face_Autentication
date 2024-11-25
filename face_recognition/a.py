import cv2
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()


def collect_training_data():
    cap = cv2.VideoCapture(0)  # Open webcam
    user_id = input("Enter user ID: ")
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} images.")


import os


def train_model():
    faces, ids = [], []
    data_path = 'dataset'
    for image_name in os.listdir(data_path):
        if image_name.endswith('.jpg'):
            img_path = os.path.join(data_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            face_id = int(image_name.split('.')[1])
            faces.append(img)
            ids.append(face_id)

    recognizer.train(faces, np.array(ids))
    recognizer.save('face_recognizer.yml')
    print("Training complete.")


def recognize_faces():
    recognizer.read('face_recognizer.yml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            id_, conf = recognizer.predict(face)
            if conf < 50:  # Adjust confidence threshold as needed
                label = f"User {id_}"
            else:
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Collect training data
collect_training_data()

# Train the recognizer
train_model()

# Perform real-time face recognition
recognize_faces()
