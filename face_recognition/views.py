# from django.shortcuts import render
# from django.core.files.storage import FileSystemStorage
#
#
# def index(request):
#     return render(request, 'index.html')
#
#
# import cv2
# import os
# from django.conf import settings
# from django.http import JsonResponse
#
#
# def detect(request):
#     if request.method == 'POST' and request.FILES['image']:
#         # Save the uploaded file
#         uploaded_file = request.FILES['image']
#         fs = FileSystemStorage()
#         file_path = fs.save(uploaded_file.name, uploaded_file)
#         file_url = fs.url(file_path)
#
#         # Perform face detection
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         img_path = os.path.join(settings.MEDIA_ROOT, file_path)
#         img = cv2.imread(img_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
#
#         # Draw rectangles around detected faces
#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#         # Save the processed image
#         processed_img_path = os.path.join(settings.MEDIA_ROOT, f"processed_{uploaded_file.name}")
#         cv2.imwrite(processed_img_path, img)
#
#         return JsonResponse({'status': 'success', 'processed_url': fs.url(f"processed_{uploaded_file.name}")})
#     return JsonResponse({'status': 'failed'})


from django.shortcuts import render, redirect
from .forms import UserForm
from django.core.files.storage import FileSystemStorage
import cv2
import os
from .models import User


def register(request):
    if request.method == 'POST':
        form = UserForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the user's data and their face image
            user = form.save()

            # Perform face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            img_path = user.image.path
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = gray[y:y + h, x:x + w]
                # Save the detected face for recognition training
                face_path = os.path.join('media', 'faces', f'{user.username}.jpg')
                cv2.imwrite(face_path, face)
                return redirect('success')  # Redirect to success page
            else:
                user.delete()
                return render(request, 'register.html', {
                    'form': form,
                    'error': 'No face or multiple faces detected. Please upload a valid image.',
                })

    else:
        form = UserForm()
    return render(request, 'register.html', {'form': form})


from django.shortcuts import render, redirect
from .forms import UserForm
from django.core.files.storage import FileSystemStorage
import cv2
import os

# from django.shortcuts import render, redirect
# from .forms import UserForm
# from .models import User
# import cv2
# import os
#
# def register(request):
#     if request.method == 'POST':
#         # Handle the form submission for username
#         form = UserForm(request.POST, request.FILES)
#
#         # Check if the "Capture from Webcam" button was clicked
#         if 'capture' in request.POST:
#             # Open the webcam for live capture
#             cap = cv2.VideoCapture(0)
#             ret, frame = cap.read()
#             cap.release()
#
#             if ret:
#                 # Save the captured image temporarily
#                 img_path = os.path.join('media', 'captured_image.jpg')
#                 cv2.imwrite(img_path, frame)
#
#                 # Display the captured image on the form
#                 return render(request, 'register.html', {
#                     'form': form,
#                     'captured_image': img_path,
#                 })
#
#         # Handle regular form submission
#         if form.is_valid():
#             # Save user data
#             user = form.save(commit=False)
#             # Use captured image instead of uploaded image
#             if 'captured_image' in request.POST:
#                 user.image.name = 'faces/' + user.username + '.jpg'
#                 img_path = os.path.join('media', 'captured_image.jpg')
#                 os.rename(img_path, os.path.join('media', user.image.name))
#
#             user.save()
#
#             # Perform face detection
#             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#             img = cv2.imread(user.image.path)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
#
#             if len(faces) == 1:
#                 x, y, w, h = faces[0]
#                 face = gray[y:y + h, x:x + w]
#
#                 # Save the detected face for recognition training
#                 face_path = os.path.join('media', 'faces', f'{user.username}.jpg')
#                 cv2.imwrite(face_path, face)
#                 return redirect('success')
#             else:
#                 user.delete()
#                 return render(request, 'register.html', {
#                     'form': form,
#                     'error': 'No face or multiple faces detected. Please try again.',
#                 })
#
#     else:
#         form = UserForm()
#
#     return render(request, 'register.html', {'form': form})


import base64
import cv2
import numpy as np
import os
from django.http import JsonResponse
from django.shortcuts import render
from .models import User

import os
import cv2
import base64
import json
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from .models import User  # Assuming you have a User model

def register(request):
    if request.method == 'POST':
        try:
            # Parse JSON body
            data = json.loads(request.body)
            image_base64 = data.get("image")

            if not image_base64:
                return JsonResponse({"success": False, "error": "No image data received."})

            # Decode the base64 image
            image_data = base64.b64decode(image_base64.split(",")[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Perform face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = gray[y:y + h, x:x + w]

                # Create directory if it doesn't exist
                face_dir = os.path.join('media', 'faces')
                os.makedirs(face_dir, exist_ok=True)

                # Save face image
                username = f"user_{User.objects.count() + 1}"
                face_path = os.path.join(face_dir, f'{username}.jpg')
                cv2.imwrite(face_path, face)

                # Save user to database
                user = User.objects.create(username=username, image=f'faces/{username}.jpg')
                return JsonResponse({"success": True})
            else:
                return JsonResponse({"success": False, "error": "No face or multiple faces detected."})

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

    return render(request, 'register.html')

# def register(request):
#     if request.method == 'POST':
#         try:
#             # Parse the JSON body to get the base64 image
#             import json
#             data = json.loads(request.body)
#             image_base64 = data.get("image")
#
#             if not image_base64:
#                 return JsonResponse({"success": False, "error": "No image data received."})
#
#             # Decode the base64 image
#             image_data = base64.b64decode(image_base64.split(",")[1])
#             np_arr = np.frombuffer(image_data, np.uint8)
#             img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
#             # Perform face detection
#             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
#
#             if len(faces) == 1:
#                 x, y, w, h = faces[0]
#                 face = gray[y:y + h, x:x + w]
#
#                 # Save the face image
#                 username = f"user_{User.objects.count() + 1}"
#                 face_path = os.path.join('media', 'faces', f'{username}.jpg')
#                 cv2.imwrite(face_path, face)
#
#                 # Save the user to the database
#                 user = User.objects.create(username=username, image=f'faces/{username}.jpg')
#
#                 return JsonResponse({"success": True})
#             else:
#                 return JsonResponse({"success": False, "error": "No face or multiple faces detecteds."})
#
#         except Exception as e:
#             return JsonResponse({"success": False, "error": str(e)})
#
#     return render(request, 'register.html')
#

import numpy as np

# def recognize(request):
#     if request.method == 'POST':
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#
#         # Train recognizer on registered faces
#         faces, labels = [], []
#         users = User.objects.all()
#         for user in users:
#             face_path = os.path.join('media', 'faces', f'{user.username}.jpg')
#             img = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
#             faces.append(np.array(img, dtype='uint8'))
#             labels.append(user.id)
#
#         recognizer.train(faces, np.array(labels))
#
#         # Capture image for recognition
#         cap = cv2.VideoCapture(0)
#         ret, frame = cap.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
#
#         for (x, y, w, h) in faces:
#             face = gray[y:y + h, x:x + w]
#             label, confidence = recognizer.predict(face)
#             if confidence < 50:  # Adjust confidence threshold as needed
#                 user = User.objects.get(id=label)
#                 result = f"Recognized: {user.username} (Confidence: {confidence})"
#             else:
#                 result = "Unknown face"
#
#         cap.release()
#         return render(request, 'recognize.html', {'result': result})
#     return render(request, 'recognize.html')


import base64
import cv2

import os
from django.http import JsonResponse
from django.shortcuts import render
from .models import User


def recognize(request):
    if request.method == 'POST':
        try:
            # Parse the JSON body to get the base64 image
            import json
            data = json.loads(request.body)
            image_base64 = data.get("image")

            if not image_base64:
                return JsonResponse({"success": False, "error": "No image data received."})

            # Decode the base64 image
            image_data = base64.b64decode(image_base64.split(",")[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Perform face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            if len(faces) > 0:
                # Initialize the recognizer
                recognizer = cv2.face.LBPHFaceRecognizer_create()

                # Train the recognizer on registered faces
                faces_db, labels_db = [], []
                users = User.objects.all()
                for user in users:
                    face_path = os.path.join('media', 'faces', f'{user.username}.jpg')
                    img = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
                    faces_db.append(np.array(img, dtype='uint8'))
                    labels_db.append(user.id)

                recognizer.train(faces_db, np.array(labels_db))

                # Try to recognize the face
                recognized = False
                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    label, confidence = recognizer.predict(face)
                    print(label,confidence,'sgrubgiurg')
                    if confidence > 50:  # Adjust confidence threshold as needed
                        user = User.objects.get(id=label)
                        return JsonResponse({"success": True, "user": user.username})
                    else:
                        recognized = False

                if not recognized:
                    return JsonResponse({"success": False, "error": "Unknown face or poor recognition."})

            else:
                return JsonResponse({"success": False, "error": "No face detected."})

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

    return render(request, 'recognize.html')


from django.shortcuts import render


def success(request):
    return render(request, 'success.html')
