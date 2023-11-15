import PIL
from flask import Flask, Response, render_template, request, redirect
import cv2, numpy, os
from cv2 import face
from PIL import Image
import base64
from webcam import Webcam
from nhandien import Recognizer
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

def read_from_webcam():
    webcam=Webcam()
    recog=Recognizer()
    while True:
        image=next(webcam.get_frame())
        
        image=recog.nhandien(image)

        yield (b'Content-Type: image/png\r\n\r\n' + image + b'\r\n--frame\r\n')

@app.route('/image_feed')
def image_feed():
    return Response(read_from_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False,host='127.0.0.1', port='8080')