import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture('fight.avi')
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=15, detectShadows=True)
    while True:
        _, frame = cap.read()
        
        mogmask = fgbg.apply(frame)
        # cv2.imshow('MOG', mogmask)
        blurmask = cv2.medianBlur(mogmask, 5)
        # cv2.imshow('Median Blur', blurmask)
        blurmask = cv2.bilateralFilter(blurmask, 9, 75, 75)
        # cv2.imshow('Bilateral Filter', blurmask)
        blurmask = cv2.GaussianBlur(blurmask, (13, 13), 5)
        cv2.imshow('Gaussian Blur', blurmask)

        pass

        k = cv2.waitKey(30)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
