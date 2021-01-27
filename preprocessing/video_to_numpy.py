import cv2  
import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm


def generate_numpy(source_path, target_path):
    for category in ['train', 'val']:
        for label in ['Fight', 'NonFight']:
            input_path = os.path.join(source_path, category, label)
            save_path = os.path.join(target_path, category, label)
            save_to_numpy(file_dir=input_path, save_dir=save_path)


def get_optical_flow(video):
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img,(224,224,1)))

    flows = []
    for i in range(0,len(video)-1):
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
        flows.append(flow)
        
    flows.append(np.zeros((224,224,2)))
      
    return np.array(flows, dtype=np.float32)


def convert_to_numpy(file_path, resize=(224,224)):
    cap = cv2.VideoCapture(file_path)
    len_frames = int(cap.get(7))

    try:
        frames = []
        for i in range(len_frames-1):
            _, frame = cap.read()
            frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224,224,3))
            frames.append(frame)   
    except:
        print('Error: ', file_path, len_frames, i)
    finally:
        frames = np.array(frames)
        cap.release()
            
    flows = get_optical_flow(frames)
    
    result = np.zeros((len(flows), 224, 224, 5))
    result[...,:3] = frames
    result[...,3:] = flows
    
    return result


def save_to_numpy(file_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        video_name = v.split('.')[0]
        video_path = os.path.join(file_dir, v)
        save_path = os.path.join(save_dir, video_name + '.npy') 
        data = convert_to_numpy(file_path=video_path, resize=(224,224))
        data = np.uint8(data)
        np.save(save_path, data)
