import cv2
import numpy as np

import novel_approach

def main():
    cap = cv2.VideoCapture('datasets/fight.avi')

    while True:
        _, frame = cap.read()

        new_frame = novel_approach.calculate_boundaries(frame)
        cv2.imshow('Novel Preprocessed Frame', new_frame)

        pass

        k = cv2.waitKey(30)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
