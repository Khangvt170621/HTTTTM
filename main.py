import PIL
from flask import Flask, Response, render_template, request, redirect
import cv2, numpy, os
from cv2 import face
from PIL import Image

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

@app.route('/train')
def train():
    path="Datasets"

    recognizer= face.LBPHFaceRecognizer_create()
    detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids=[]
        for imagePath in imagePaths:
            try:
                PIL_img=Image.open(imagePath).convert('L')
                img_numpy=numpy.array(PIL_img,"uint8")
                id=int(os.path.split(imagePath)[-1].split(".")[1])
                faces=detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            except PIL.UnidentifiedImageError:
                print(f"Unable to identify image: {imagePath}")
                continue

        return faceSamples,ids

    print("\n INFO Đang trainning dữ liệu...")
    faces,ids=getImagesAndLabels(path)
    recognizer.train(faces,numpy.array(ids))

    recognizer.write("trainer/trainer.yml")

    print("\n INFO {0} khuôn mặt đã được trainning. Thoát".format(len(numpy.unique(ids))))

    return render_template('train.html')


if __name__ == '__main__':
    app.run(debug=False,host='127.0.0.1', port='8080')