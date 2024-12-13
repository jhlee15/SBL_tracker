import torch
import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import (StrongSort, BotSort, DeepOcSort, OcSort, ByteTrack, ImprAssocTrack, get_tracker_config, create_tracker)
import yaml
from types import SimpleNamespace

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# train/tracking parameter config
file_path = 'config.yaml' 
config = load_yaml(file_path)

# detection model load (torch, tensorrt) 
detector = YOLO(config['yolo_model'])

# Initialize the tracker
# define bytetrack  
tracker = ByteTrack(track_thresh = config['track_thresh'],
                    match_thresh = config['match_thresh'],
                    track_buffer = config['track_buffer'],
                    frame_rate = config['frame_rate'],
                    per_class = config['per_class'],)
    

# Open the video file
vid = cv2.VideoCapture(config['source'])

# categories
cat = {0:"person",1:"hat",2:"glasses",3:"face_shield",4:"mask",5:"gloves",6:"suit",7:"boots",8:"integ"}

# video inference
while True:
    
    ret, frame = vid.read()

    if not ret:
        break

    # Perform detection ultralytics yolov11
    with torch.no_grad():
        results = detector.predict(
            source= frame,
            conf= config['conf'],
            iou= config['iou'],
            agnostic_nms= config['agnostic_nms'],
            show= config['show'],
            stream= config['stream'],
            device= config['device'],
            show_conf= config['show_conf'],
            save_txt= config['save_txt'],
            show_labels= config['show_labels'],
            save= config['save'],
            verbose= config['verbose'],
            project= config['project'],
            name= config['name'],
            classes= config['classes'],
            imgsz= config['imgsz'],
            vid_stride= config['vid_stride'],
            line_width= config['line_width'])

    dets = []
    # append detection results to dets
    for result in results:
        for i in range(len(result.boxes)) :
            bbox = result.boxes.xyxy[i].cpu().numpy()
            label = result.boxes.cls[i].cpu().numpy()
            conf = result.boxes.conf[i].cpu().numpy()
            dets.append([*bbox, conf, label])
            
    dets = np.array(dets)
    # Update the tracker
    tracking_data = []
    track_result = tracker.update(dets, frame)
    tracking_data.append([{"tracking_data":track_result}])
    
    # equipment results per frame
    equip_info = tracker.equip_info
    
    # detection results per frame
    det_result = tracker.det_result
    
    # print(f"tracking_data:{tracking_data}")
    print(f"equip_info : {tracker.equip_info}")
    print(f"det_result : {tracker.det_result}")

vid.release()