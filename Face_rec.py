import face_recognition
import cv2
import numpy as np
import os


def face_encoding():
    encoded_face = {}
    for dirpath, dirname, filename in os.walk("./faces"):
        for x in filename:
            if x.endswith(".jpg"):
                face = face_recognition.load_image_file("faces/" + x)
                encoding = face_recognition.face_encodings(face)[0]
                encoded_face[x.split(".")[0]] = encoding
    return encoded_face


faces = face_encoding()
known_encodings = list(faces.values())
known_face_names = list(faces.keys())


image = cv2.imread("test.jpg")
face_locations = face_recognition.face_locations(image)
unknown_encoding = face_recognition.face_encodings(image)
face_names = []

for face_search in unknown_encoding:
    match = face_recognition.compare_faces(known_encodings, face_search)
    name = "unknown Face"
    face_distances = face_recognition.face_distance(known_encodings, face_search)
    best_match = np.argmin(face_distances)

    if match[best_match]:
        name = known_face_names[best_match]
    face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # draw a rectangle around the face
        cv2.rectangle(image, (left-30, top-30), (right + 30, bottom + 30), (168, 164, 50), 3)
        # draw a name tag below the face
        cv2.rectangle(image, (left-30, bottom-15), (right+30, bottom+30), (168, 164, 50), cv2.FILLED)
        font = cv2.FONT_ITALIC
        cv2.putText(image, name, (left-20, bottom+15), font, 1.0, (255, 255, 255), 1)

while True:
    cv2.imshow("Face recognition Model", image)
    # press 'Esc' key in keyboard to escape
    key = cv2.waitKey(0)
    if key == 27:
        break
