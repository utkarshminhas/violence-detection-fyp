import cv2
import numpy as np

fgbg = cv2.createBackgroundSubtractorMOG2(
    varThreshold=15,
    detectShadows=True
)

def get_mask(frame):
    mog_mask = fgbg.apply(frame)
    median_blur_mask = cv2.medianBlur(mog_mask, 5)
    bilateral_filter_mask = cv2.bilateralFilter(median_blur_mask, 9, 75, 75)
    gaussian_blur_mask = cv2.GaussianBlur(bilateral_filter_mask, (13, 13), 5)

    return gaussian_blur_mask

def calculate_boundaries(frame):
    masked_frame = get_mask(frame)
    return masked_frame


def display_preprocessed_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()

        if not success or frame is None:
            break

        new_frame = get_mask(frame)
        cv2.imshow('Novel Preprocessed Frame', new_frame)

        k = cv2.waitKey(30)
        if k == 27 or k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
