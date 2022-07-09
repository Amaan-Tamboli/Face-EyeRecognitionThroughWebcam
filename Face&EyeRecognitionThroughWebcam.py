
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


while True:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 9)
   

    eyes = eye_cascade.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 30)


    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)


    for x,y,w,h in eyes:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

        
    cv2.imshow('LIVE FACE AND EYE RECOGNITION', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
     

    
    
