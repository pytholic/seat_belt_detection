
from os.path import dirname, abspath

import cv2
from contextlib import contextmanager
import numpy as np
import logging
import time
from collections import deque

VIDEO = "./videos/6.mp4"  # 2.mp4
WEIGHTS = "YOLOFI2.weights"
CONFIG = "YOLOFI.cfg"
OBJ_NAMES = "obj.names"

class BeltDetected:

    def __init__(self):
        self.belt_frames = []  # main part

    def add_belt(self, frame):
        self.belt_frames.append(frame)
  
    
@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def get_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[layer - 1] for layer in net.getUnconnectedOutLayers()]


def get_classes():
    classes = []
    with open(OBJ_NAMES, "r") as file:
        classes = [line.strip() for line in file.readlines()]
    return classes


def belt_detector(net, img, belt_detected, current_frame):
    pred = []
    blob = cv2.dnn.blobFromImage(img, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
    
    height, width, channels = img.shape
    
    start = time.time()
    
    net.setInput(blob)
    outs = net.forward(get_layers(net))
    
    end=time.time()
    
    # Calulate fps
    fps = 1 / (end - start)
    
    for out in outs: 
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                # Draw boxes for testing
                # w = int(detection[2] * width)
                # h = int(detection[3] * height)
                # x = int(center_x - w / 2)
                # y = int(center_y - h / 2)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if class_id == 0:
                    belt_detected.add_belt(current_frame)
                    pred.append('detected')
    
    # Remove redundant predictions
    pred = set(pred)
    
    return belt_detected, pred, fps


def apply_clahe(img, **kwargs):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab = list(cv2.split(lab))
    clahe = cv2.createCLAHE(**kwargs)
    lab[0] = clahe.apply(lab[0])
    lab = cv2.merge((lab[0], lab[1], lab[2]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gabor(img, **kwargs):
    g_kernel = cv2.getGaborKernel(**kwargs)
    return cv2.filter2D(img, cv2.CV_8UC3, g_kernel.sum())


def increase_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def print_text(img, text: str, org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(0,255,0), thickness=5):
    cv2.putText(img, text, org=org, fontFace=fontFace, fontScale=fontScale, color=color, thickness=thickness)


def main():
    with video_capture(VIDEO) as cap:

        predictions = deque([])
        net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        frame_id = -1
        belt_detected = BeltDetected()
        #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (2200,1000))
        while True:            
            frame = cap.read()
            frame_id += 1
            
            if not frame[0]:
                break
            img = frame[1]
            
            ### PREPROCESSING ###
            
            # Flip for passenger
            img = cv2.flip(img, 1)
            
            img = img[300:800, 1000:1500]  # [300:800, 300:1500]
            
            kernel_sharp = np.array([[-1, -1, -1],
                               [-1, 9,-1],
                               [-1, -1, -1]])
            img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_sharp)
            
            img = cv2.GaussianBlur(img, (5,5), 0)
            
            #print(img.shape)

            img = increase_brightness(img)
            img = apply_clahe(img=img, clipLimit=5, tileGridSize=(15, 15))
            img = apply_gabor(img=img, ksize=(31, 31), sigma=2.9, theta=160,
                              lambd=14.5, gamma=35, psi=50, ktype=cv2.CV_64F)
            
            ### DETECTION ###
            
            belt_detected, pred, fps = belt_detector(net, img, belt_detected, frame_id)
            
            ### RESULTS PREOCESSING ###
            
            # Show fps
            print_text(img, f"FPS: {fps:.2f}", org=(20,480), fontScale=1.5, color=(0,0,255), thickness=2)
            
            # Append results in predictions array
            if len(pred) > 0:
                predictions.appendleft("Detected")
            else:
                predictions.appendleft("Not detected")
            
            if len(predictions) > 200:
                predictions.pop()
            
            img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))  

            # Threshold logic
            cnt_on = predictions.count("Detected")
            cnt_off = predictions.count("Not detected")
            
            thres = cnt_on / (cnt_on + cnt_off)
            
            if thres > 0.5:
                print_text(img, "Belt is on", org=(100,100))
                #print("Belt is on.")
            else:
                print_text(img, "Belt is off", org=(100,100))

            print("Passenger: ", cnt_on, cnt_off)

            cv2.imshow("Image", img)

            #out.write(img)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        cap.release()    
        #out.release()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
        main()
