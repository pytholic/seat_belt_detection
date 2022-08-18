
from os.path import dirname, abspath

import cv2
import numpy as np
import logging
import time
import multiprocessing
import threading

from multiprocessing import Queue
from collections import deque
from contextlib import contextmanager


VIDEO = "./videos/1.mp4" 
WEIGHTS = "YOLOFI2.weights"
CONFIG = "YOLOFI.cfg"
OBJ_NAMES = "obj.names"
NUM_FRAMES = 200

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


def belt_detector(queue, net, img_list, belt_detected, current_frame):
    pred = [] 
    blob = cv2.dnn.blobFromImages(img_list, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
    height, width, channels = img_list[0].shape
      
    net.setInput(blob)
    outs = net.forward(get_layers(net))
     
    for out in outs: 
        for detections in out:
            temp = []
            for detection in detections:                
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    if class_id == 0:
                        belt_detected.add_belt(current_frame)
                        
                        temp.append('detected')

            # Remove redundant predictions
            temp = set(temp)
                
            if len(temp) > 0:
                pred.append("Detected")   
    print('done')
    queue.put(belt_detected, pred)
    #return belt_detected, pred

# Preprocessing utilities

def apply_clahe(img, clipLimit=5, tileGridSize=(15, 15)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    lab[0] = clahe.apply(lab[0])
    lab = cv2.merge((lab[0], lab[1], lab[2]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gabor(img, ksize=(31, 31), sigma=2.9, theta=160,
                            lambd=14.5, gamma=35, psi=50, ktype=cv2.CV_64F):
    g_kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)
    return cv2.filter2D(img, cv2.CV_8UC3, g_kernel.sum())


def increase_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def sharpen(img):
    kernel_sharp = np.array([[-1, -1, -1],
                             [-1, 9,-1],
                             [-1, -1, -1]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_sharp)
    return img


def preprocess(img):
    img = sharpen(img)
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = increase_brightness(img)
    img = apply_clahe(img)
    img = apply_gabor(img)
    return img

# Print function (for testing)
def print_text(img, text: str, org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,255,0), thickness=2):
    cv2.putText(img, text, org=org, fontFace=fontFace, fontScale=fontScale, color=color, thickness=thickness)


def inference(net, img_list, frame_id, belt_detected, passenger: str):
    """
    net: Model architecture to use for inference
    img_list: Batch of input frames
    frame_id: Current frame number
    belt_detector: BeltDetector object
    passenger: One of 'driver' or 'passenger'
    """
    
    queue = Queue()
                              
    ### DETECTION ###
    
    belt_detected, pred = belt_detector(queue, net, img_list, belt_detected, frame_id)
    
    item = queue.get()
    
    print(item)
    
    ### RESULTS PROCESSING ###

    # Count predictions
    cnt_on = len(pred)
    
    # Get ratio
    thres = cnt_on / NUM_FRAMES
    
    if thres > 0.5:
        print(f"Belt is on for {passenger}")
    else:
        print(f"Belt is off for {passenger}")
                
            
def test(num, passenger: str):
    print(f"hello {passenger}")  
    print(num)


def main():
    
    #print("Number of cpu : ", multiprocessing.cpu_count())
    
    with video_capture(VIDEO) as cap:
        img_list = []
        img_flipped_list = []
        net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        frame_id = -1
        belt_detected = BeltDetected()
        
        while True:    
            frame = cap.read()
            frame_id += 1
            
            if not frame[0]:
                break
            img = frame[1]
                        
            # Preprocessing 
            img_flipped = cv2.flip(img, 1)
            img_flipped = img_flipped[300:800, 1000:1500]
            img = img[300:800, 1000:1500]
            img_flipped = preprocess(img_flipped)
            img = preprocess(img)
            img_list.append(img)
            img_flipped_list.append(img_flipped)
                      
            if frame_id % 200 == 0 and frame_id != 0:  
                print(f"Current frame: {frame_id}")
                print('\n')

                # Inference
                # t1 = threading.Thread(target=inference, args=(net, img_flipped_list, frame_id, belt_detected), kwargs={"passenger": 'passenger'})
                # t2 = threading.Thread(target=inference, args=(net, img_list, frame_id, belt_detected), kwargs={"passenger": 'driver'})
                
                # #t1 = threading.Thread(target=test, kwargs={"passenger":'passsenger'})
                # #t2 = threading.Thread(target=test, kwargs={"passenger":'driver'})
                
                # print("***Starting inference***")
                # start = time.time()
                
                # t1.start()
                # print("Waiting for process 1")
                # t2.start()
                # print("Waiting for process 2")
                # t1.join()
                # print("Process 1 ended")
                # t2.join()
                # print("Process 2 ended")
                
                # end = time.time()


                # Inference

                p1 = multiprocessing.Process(target=inference, args=[net, img_flipped_list, frame_id, belt_detected, 'passenger']) #, kwargs={"passenger": 'passenger'})
                #p2 = multiprocessing.Process(target=inference, args=[(net,) img_list, frame_id, belt_detected)], kwargs={"passenger": 'driver'})
                
                #num=1
                #p1 = multiprocessing.Process(target=test, args=[num, 'driver']) #, kwargs={"passenger":'passsenger'})
                # p2 = multiprocessing.Process(target=test, args=[(num)], kwargs={"passenger":'driver'})
                
                print("***Starting inference***")
                start = time.time()
                
                p1.start()
                print("Waiting for process 1")
                #p2.start()
                #print("Waiting for process 2")
                p1.join()
                print("Process 1 ended")
                #p2.join()
                #print("Process 2 ended")
                
                end = time.time()
                
                
                # num=1
                
                # pool = multiprocessing.Pool()
                
                # print("***Starting inference***")
                # start = time.time()
                
                # #pool.starmap(inference, [(net, img_flipped_list, frame_id, belt_detected, 'passenger')])
                # #pool.starmap(test, [(num, 'driver')])
                # #pool.starmap(test, [(num, 'passenger')])

                # pool.close()
                # pool.join()
                
                # end = time.time()
                
                img_list.clear()
                img_flipped_list.clear()
                
                print("\n")
                
                # Calulate total time
                total_time = end - start
                print(f"Total inference time: {total_time}")
            
                #break
            #run_once = False
        
            # img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2)) 
            
            # cv2.imshow("Image", img)
                        
            # key = cv2.waitKey(1)
            # if key == 27:
            #     break
            
        cap.release()    
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
        main()
