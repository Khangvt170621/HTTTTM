import cv2, sys, numpy, os, PIL
from cv2 import face
from PIL import Image
from datetime import datetime

class Recognizer():
    def nhandien(self, img):
        recognizer= cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer/trainer.yml")
        detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        font=cv2.FONT_HERSHEY_SIMPLEX

        id=0

        names=['Dang Bao Khang', 'Nguyen Duc Anh Tho','Nguyen Huu Thien',"4"]
        webcam=cv2.VideoCapture(0)
        
        ret, img= webcam.read()
        # img=cv2.flip(img,-1)
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(
            gray,
            scaleFactor=1.2,                
            minNeighbors=5,
        )

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id,confidence=recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence<100):
                id,confidence=recognizer.predict(gray[y:y+h,x:x+w])
                id=names[id]
                confidence="{0}%".format(round(100-confidence))
            else:
                id="unknown"
                confidence="{0}%".format(round(100-confidence))
            cv2.putText(img, str(id),(x+5,y-5), font,1,(0,0,255),2)
            cv2.putText(img, str(confidence), (x+5,y+h-5),font,1,(0,0,255),1)

        font=cv2.FONT_HERSHEY_SIMPLEX
        org= (50,50)
        fontScale=1
        color=(0,255,0)
        thickness=2

        img=cv2.putText(img, datetime.now().strftime("%H:%M:%S"),org,font,
                            fontScale, color, thickness, cv2.LINE_AA)    
        return cv2.imencode('.png', img)[1].tobytes()
    
