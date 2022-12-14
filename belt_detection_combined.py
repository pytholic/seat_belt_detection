
from os.path import dirname, abspath

import cv2
from contextlib import contextmanager
import numpy as np
import logging
import time

VIDEO = "./videos/2.mp4"  # 2.mp4
WEIGHTS = "YOLOFI2.weights"
CONFIG = "YOLOFI.cfg"
OBJ_NAMES = "obj.names"
SAVE_PATH = dirname(dirname(abspath(__file__))) + "/"


logging.basicConfig(level=logging.INFO)

class BeltDetected:
    # list of frames (ids) where the belt part was detected as closed
    # first frame has id 0
    def __init__(self):
        self.belt_frames = []  # main part
        self.belt_corner_frames = []  # corner part
        self.result = None

    def add_belt(self, frame):
        self.belt_frames.append(frame)

    def add_corner_belt(self, frame):
        self.belt_corner_frames.append(frame)
    
@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    #start_frame_number = 500
    #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
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
    net.setInput(blob)

    height, width, channels = img.shape

    outs = net.forward(get_layers(net))
    for out in outs:
        for detection in out:
            #print(detection)
            scores = detection[5:]
            class_id = np.argmax(scores)
            #print(class_id)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # if class_id == 1:
                #     belt_detected.add_corner_belt(current_frame)
                #     pred.append("detected")
                if class_id == 0:
                    belt_detected.add_belt(current_frame)
                    pred.append("detected")

    return belt_detected, pred


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

def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

def main():
    with video_capture(VIDEO) as cap:

        predictions = []
        net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        frame_id = -1
        belt_detected = BeltDetected()
        #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (2048,1024))
        while True:
            frame = cap.read()
            frame_id += 1
            
            if not frame[0]:
                break
            #img = frame[1][:, 50: -50]
            img = frame[1]
            
            img = img[300:800, 400:1500]  # [300:800, 300:1500]
            print(img.shape)     
            #img = cv2.flip(img, 0)
            
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.equalizeHist(img)
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # r_image, g_image, b_image = cv2.split(img)
            # r_image_eq = cv2.equalizeHist(r_image)
            # g_image_eq = cv2.equalizeHist(g_image)
            # b_image_eq = cv2.equalizeHist(b_image)
            # img = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
            
            img = cv2.GaussianBlur(img, (9,9), 0)
            #img = cv2.medianBlur((img), 15)

            # kernel_sharp = np.array([[0, -1, 0],
            #                    [-1, 5,-1],
            #                    [0, -1, 0]])
            kernel_sharp = np.array([[-1, -1, -1],
                               [-1, 9,-1],
                               [-1, -1, -1]])
            img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_sharp)
            
            #print(img.shape)

            img = increase_brightness(img)
            #img = apply_clahe(img=img, clipLimit=5, tileGridSize=(5, 5))
            img = apply_gabor(img=img, ksize=(31, 31), sigma=2.9, theta=160,
                              lambd=14.5, gamma=35, psi=50, ktype=cv2.CV_64F)
            
            # Extra preprocessing
            
            # img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
            # img = adjust_gamma(img, gamma=2.0)
            #img = cv2.detailEnhance(img, sigma_s=25, sigma_r=0.15)  # 0.15
            
            belt_detected, pred = belt_detector(net, img, belt_detected, frame_id)
            
            # Append results in predictions array
            if len(pred) > 0:
                predictions.insert(0, "Detected")
            else:
                predictions.insert(0, "Not detected")
                
            # Pop last element when size is greater than 50
            # this way we restrict our predictions length 
            # to recent 200 frames
            
            if len(predictions) > 200:
                predictions.pop()
                
            img = cv2.resize(img, (2048, 1024))  

            #out.write(img)

            # Threshold logic
            cnt_on = predictions.count("Detected")
            cnt_off = predictions.count("Not detected")
            thres = cnt_on / (cnt_on + cnt_off)
            print(cnt_on, cnt_off)
            
            if thres > 0.5:
                cv2.putText(img, "Belt is on", org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(0,255,0), thickness=5)
                #print("Belt is on.")
            else:
                cv2.putText(img, "Belt is off", org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(0,255,0), thickness=5)
                #print("Belt is off.")

            cv2.imshow("Image", img)

            key = cv2.waitKey(1)
            if key == 27:
                break
        
        cap.release()    
        #out.release()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
        main()
