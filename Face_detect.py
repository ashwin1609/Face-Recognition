import numpy as np
import cv2

img = cv2.imread('pic5.jpg')
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
body = cv2.CascadeClassifier('haarcascade_fullbody.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

full_body = body.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)
for (p,q,r,s) in full_body:
    # draw a rectangle around the body
     img = cv2.rectangle(img,(p,q),(p+r,q+s),(250,180,25),2)
     img_gray = gray[q:q+s, p:p+r]
     img_color = img[q:q+s, p:p+r]
     faces = face.detectMultiScale(img_gray)
     for (a,b,c,d) in faces:
         # Find the face within the body and draw a rectangle
        cv2.rectangle(img_color,(a,b),(a+c,b+d),(0,255,0),2)
        roi_gray = gray[b:b+d, a:a+c]
        roi_color = img[b:b+d, a:a+c]
        eyes = eye.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            # Find the eyes within the face and draw a rectangle
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

cv2.imshow('ronaldo',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
