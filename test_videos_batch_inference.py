
from os.path import dirname, abspath

import cv2
from contextlib import contextmanager
import numpy as np
import logging
import time
from collections import deque

VIDEO = "./videos/2.mp4"  # 2.mp4
WEIGHTS = "YOLOFI2.weights"
CONFIG = "YOLOFI.cfg"
OBJ_NAMES = "obj.names"
SAVE_PATH = dirname(dirname(abspath(__file__))) + "/"
NUM_FRAMES = 200


logging.basicConfig(level=logging.INFO)

class BeltDetected:
    # list of frames (ids) where the belt part was detected as closed
    # first frame has id 0
    def __init__(self):
        self.belt_frames = []  # main part
        self.belt_corner_frames = []  # corner part

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


def belt_detector(net, img_list, belt_detected, current_frame):
    
    pred_left = []
    pred_right = []
        
    blob = cv2.dnn.blobFromImages(img_list, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
    
    height, width, channels = img_list[0].shape
    
    mid_x = width / 2
    mid_y = height / 2
    
    start = time.time()
    
    net.setInput(blob)
    outs = net.forward(get_layers(net))
    
    end=time.time()
    
    # Calulate fps
    fps = 1 / (end - start)
    
    for out in outs: 
        for detections in out:
            left = []
            right = []
            for detection in detections:                
                scores = detection[5:]
                class_id = np.argmax(scores)
                #print(class_id)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    # w = int(detection[2] * width)
                    # h = int(detection[3] * height)
                    # x = int(center_x - w / 2)
                    # y = int(center_y - h / 2)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # if class_id == 1:
                    #     belt_detected.add_corner_belt(current_frame)
                    #     pred.append("detected")
                    if class_id == 0:
                        belt_detected.add_belt(current_frame)
                        
                        if center_x < mid_x:
                            left.append("detected left")
                        elif center_x > mid_x:
                            right.append("detected right")
                        else:
                            pass

            left = set(left)
            right = set(right)
                
            if len(left) > 0:
                pred_left.append("Detected Left")

            if len(right) > 0:
                pred_right.append("Detected Right")    
        
    return belt_detected, pred_left, pred_right, fps


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

def print_text(img, text: str, org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,255,0), thickness=2):
    cv2.putText(img, text, org=org, fontFace=fontFace, fontScale=fontScale, color=color, thickness=thickness)

def main():
    with video_capture(VIDEO) as cap:
        img_list = []
        net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        frame_id = -1
        belt_detected = BeltDetected()
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (2200,1000))
        while True:    
            frame = cap.read()
            frame_id += 1
            
            if not frame[0]:
                break
            img = frame[1]
            
            ### PREPROCESSING ###
            
            img = img[300:800, 400:1500]  # [300:800, 300:1500]

            kernel_sharp = np.array([[-1, -1, -1],
                                        [-1, 9,-1],
                                        [-1, -1, -1]])
            img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_sharp)
            img = cv2.GaussianBlur(img, (5,5), 0)
            

            img = increase_brightness(img)
            #img = apply_clahe(img=img, clipLimit=5, tileGridSize=(5, 5))
            img = apply_gabor(img=img, ksize=(31, 31), sigma=2.9, theta=160,
                            lambd=14.5, gamma=35, psi=50, ktype=cv2.CV_64F)
            
            img_list.append(img)
            
            if (len(img_list) % 200) == 0:
                                    
                ### DETECTION ###
                belt_detected, pred_left, pred_right, fps = belt_detector(net, img_list, belt_detected, frame_id)
            
                print(len(pred_left))
                print(len(pred_right))
                        
                ### RESULTS PROCESSING ###
            
                # Count predictions
                cnt_on_left = len(pred_left)
                cnt_on_right = len(pred_right)
                
                # Get ratio
                thres_left = cnt_on_left / NUM_FRAMES
                thres_right = cnt_on_right / NUM_FRAMES
                
                if thres_left > 0.5:
                    print_text(img, "Left belt is on", org=(100,100))
                    #print("Belt is on.")
                else:
                    print_text(img, "Left belt is off", org=(100,100))
                    #print("Belt is off.")
                                
                if thres_right > 0.5:
                    print_text(img, "Right belt is on", org=(600,100))
                    #print("Belt is on.")
                else:
                    print_text(img, "Right belt is off", org=(600,100))
                    #print("Belt is off.")
            
                img_list.clear()
                
        #     # Show fps
        #     # print_text(img, f"FPS: {fps:.2f}", org=(20,480), fontScale=1.5, color=(0,0,255), thickness=2)
            
        #     # print("Left Passenger: ", cnt_on_left, cnt_off_left,"\t\t", "Right Passenger: ", cnt_on_right, cnt_off_right)
            
            img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2)) 
            
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
