import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
app = Flask(__name__)
maxSpeed = 10

def preProcess(img):
    img = img[60:135, :, :]  # Crop unnecessary top and bottom parts
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur for noise reduction
    img = cv2.resize(img, (200, 66))  # Resize image to a fixed size
    img = img / 255  # Normalize pixel values between 0 and 1
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])  # Add an extra dimension for the model (batch size of 1)
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print(f'{steering} {throttle} {speed}')
    sendControl(steering, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)  # Send initial control commands (0 steering and neutral throttle)

def sendControl(steering, throttle):
    sio.emit('steer', data={'steering_angle': str(steering), 'throttle': str(throttle)})

if __name__ == '__main__':
    # Load the model with the custom_objects parameter
    model = load_model('model.h5')
    app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)  # Start the server on port 4567
