import cv2
import numpy as np
import os
import sys
import math
from tqdm import tqdm

import novel_approach
import paths

def main():
    source_path = paths.source_path
    target_path = paths.target_path
    
    for category in ['train', 'val']:
        for label in ['Fight', 'NonFight']:
            input_path = os.path.join(source_path, category, label)
            save_path = os.path.join(target_path, category, label)
            save_to_frames(input_path=input_path, save_dir=save_path)

def save_to_frames(input_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    videos = os.listdir(input_path)

    for v in tqdm(videos):
        video_name = v.split('.')[0]
        video_path = os.path.join(input_path, v)

        target_base_name = os.path.join(save_dir, video_name)

        cap = cv2.VideoCapture(video_path)
        framerate = int(cap.get(5))
        
        while True:
            frame_number = int(cap.get(1))
            success, frame = cap.read()

            if not success or frame is None:
                break
            
            cv2.imshow('Original Frame', frame)
            masked_frame = novel_approach.get_mask(frame)
            # frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)

            if (frame_number % framerate == 0):
                exporting_frame_number = int(frame_number / framerate)

                save_path = target_base_name + '_frame' + str(exporting_frame_number) + '.jpg'
                cv2.imwrite(save_path, masked_frame)
                
                k = cv2.waitKey(30)

                if k == 27 or k == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
