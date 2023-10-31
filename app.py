from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import json

# from PIL import ImageGrab
app = Flask(__name__)

image_folder = 'Images'
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
# # img = captureScreen()
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# # print(faceDis)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
# # print(name)
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             markAttendance(name)

#     cv2.imshow('Webcam', img)
#     cv2.waitKey(1)

# Create an empty list to store recognized names
recognized_names = []

image_encodings = {}
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(image_path)
        image_encoding = face_recognition.face_encodings(image)[0]
        image_encodings[filename] = image_encoding
        
def match_face(input_image):
    if not image_encodings:
        return None

    input_image = face_recognition.load_image_file(input_image)
    input_encoding = face_recognition.face_encodings(input_image)

    if not input_encoding:
        return None

    for filename, encoding in image_encodings.items():
        matches = face_recognition.compare_faces([encoding], input_encoding[0])
        if matches[0]:
            return filename

    return None       
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    name = request.form['name']
    match = match_face(request.files['image'])

    if match:
        recognized_names.append(match)
        return json.dumps({'name': match, 'match': True})
    else:
        return json.dumps({'name': '', 'match': False})

# SSE route to send recognized names to the UI
def generate():
    while True:
        if recognized_names:
            name = recognized_names.pop(0)
            yield f"data: {json.dumps({'name': name})}\n\n"
        else:
            yield f"data: {json.dumps({'name': ''})}\n\n"

@app.route('/sse')
def sse():
    return Response(generate(), content_type='text/event-stream')

def match_face(input_image):
    input_image = face_recognition.load_image_file(input_image)
    input_encoding = face_recognition.face_encodings(input_image)

    if not input_encoding:
        return None

    for filename, encoding in image_encodings.items():
        matches = face_recognition.compare_faces([encoding], input_encoding[0])
        if matches[0]:
            return filename

if __name__ == "__main__":
    app.run(debug=True)
