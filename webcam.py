import cv2
from datetime import datetime

class Webcam():
    def __init__(self):
        self.vid=cv2.VideoCapture(0)

    def get_frame(self):
        if not self.vid.isOpened():
            return
        
        while True:
            _, img=self.vid.read()

            font=cv2.FONT_HERSHEY_SIMPLEX
            org= (50,50)
            fontScale=1
            color=(0,255,0)
            thickness=2

            img=cv2.putText(img, datetime.now().strftime("%H:%M:%S"),org,font,
                            fontScale, color, thickness, cv2.LINE_AA)
            
            yield img #(cv2.imencode('.png', img)[1].tobytes())
