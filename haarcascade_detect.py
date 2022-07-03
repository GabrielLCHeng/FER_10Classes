import cv2
import os

CASC_PATH = './media/haarcascade_frontalface_default.xml'
# CASC_PATH = 'haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def face_detect(image):
    # CASC_PATH = './media/haarcascade_frontalface_default.xml'
    # cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    if not len(faces) > 0:
        print('Not detected')
        return []
    else:
        for x,y,w,h in faces:
            cv2.rectangle(img=image, pt1=(x,y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
            # to_ndarray(format="bgr24")
    return faces