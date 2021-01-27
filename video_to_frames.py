import cv2
import numpy as np
import os
import novel_approach
import paths
import sys
import math

def main():
    cap = cv2.VideoCapture("datasets/testvid.avi")
    framerate = cap.get(5)
    print()
    path=os.getcwd()
    print(os.getcwd())
    # sys.exit()
    # os.mkdir("images")
    while True:
        frame_number = cap.get(1)
        success, frame = cap.read()
        if frame is not None:
            cv2.imshow('original frame', frame)
            # frame=cv2.resize(frame,(224,224), interpolation = cv2.INTER_AREA)
            frame = novel_approach.calculate_boundaries(frame)
        if (success != True):
            print("breaking cuz no success")
            break
        if (frame_number % math.floor(framerate) == 2):
            print("\n Frame number: ",frame_number)
            print("trying to save frame number ",str(int(frame_number / math.floor(framerate))+1))
            filename = "datasets/images/image_" + str(int(frame_number / math.floor(framerate))+1) + ".jpg"
            cv2.imwrite(filename,frame)
            print(filename)
       
        k = cv2.waitKey(30)
        if k == 27 or k==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
