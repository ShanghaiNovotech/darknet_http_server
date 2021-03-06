from ctypes import *
import sys
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

process_fps = 0.3


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def detStats(detections):
    dets=[]
    dets = [item[0] for item in detections]
    freqs = {i.decode():dets.count(i) for i in set(dets)}
    return freqs

def cvDrawSummary(freqs, img):
    for key, value in freqs.items():
        if key=="person":
            print_str= "PERSON " + str(value)
            cv2.putText(img, print_str, (5, 20) ,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 0], 2)
    return img


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def print2DMatrix(header, matrix):
    print(''.join(['{:10}'.format(item) for item in header]))
    print('\n'.join([''.join(['{:10}'.format(str(item)) for item in row]) for row in matrix]))

def get_film_det_csv(film_det_freqs):
    film_det_keys =set([item for sublist in film_det_freqs for item in sublist])
    film_det_freq_csv=[]
    for i in range(len(film_det_freqs)):
        film_det_freq_csv.append([])
        for key in film_det_keys:
            film_det_freq_csv[i].append(film_det_freqs[i].get(key,0))
            
    print2DMatrix(film_det_keys, film_det_freq_csv)
    print(getCSV(film_det_keys, film_det_freq_csv))
    return film_det_freq_csv

def getCSV(header, matrix):
    return ",".join(header) +"\n"+ "\n".join([",".join([str(itm) for itm in row]) for row in matrix])

netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(sys.argv[1])
    cap.set(3, 1920)
    cap.set(4, 1080)
    #out =  cv2.VideoWriter('output.mp4', 0x00000021, 15.0, (1920,1080))
    out = cv2.VideoWriter( "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (darknet.network_width(netMain), darknet.network_height(netMain)))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)
 #   fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    #print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    processed_time_sec= 0

    film_det_freqs=[]
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if not ret:
            break
        time_sec=(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)
        
        #skip frames 
        if processed_time_sec < time_sec:
            processed_time_sec += 1/process_fps
        else:
            continue

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        freqs = detStats(detections)
        image = cvDrawSummary(freqs, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out.write(image)
        print("File: "+ str(round(time_sec,2))+" sec. Frame proc time: " + str(round(time.time()-prev_time,3)) + " sec.")
        film_det_freqs.append(freqs)
        cv2.imshow('Demo', image)
        cv2.waitKey(3)

    get_film_det_csv(film_det_freqs)
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()

