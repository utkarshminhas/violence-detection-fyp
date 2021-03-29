import cv2
import numpy as np
from functools import cmp_to_key


# video_001 -> [subvideo_01, subvideo_02, ...]
def holi_approach(video):
  subvideos = []

  # Do somthing yaha pe

  return subvideos

def resize_contour(current_size, aspect_ratio = (1, 1)):
  (x, y, w, h) = current_size

  if w / h > aspect_ratio[0] / aspect_ratio[1]:
    factor = w / aspect_ratio[0]
    delta = factor * aspect_ratio[1] - h
    y -= delta / 2
    h = factor * aspect_ratio[1]
  else:
    factor = h / aspect_ratio[1]
    delta = factor * aspect_ratio[0] - w
    x -= delta / 2
    w = factor * aspect_ratio[0]

  x = int(x)
  y = int(y)
  w = int(w)
  h = int(h)

  return (x, y, w, h)

def display_video(video_path):
  cap = cv2.VideoCapture(video_path)

  _, previous_frame = cap.read()
  _, current_frame = cap.read()

  while cap.isOpened():
    diff = cv2.absdiff(previous_frame, current_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    nominated_contours = []

    for contour in contours:
      (x, y, w, h) = resize_contour(cv2.boundingRect(contour), (1, 1))

      if cv2.contourArea(contour) < 900:
        continue

      nominated_contours.append((x, y, w, h))

    winning_contours = []
    nominated_contours = sorted(nominated_contours, key=cmp_to_key(lambda item1, item2: item2[2] * item2[3] - item1[2] * item1[3]))

    for i in range(len(nominated_contours)):
      flag = True

      for j in range(i):
        (ix, iy, iw, ih) = nominated_contours[i]
        (jx, jy, jw, jh) = nominated_contours[j]

        if not (iy >= jy + jh or iy + ih <= jy or ix >= jx + jw or ix + iw <= jx):
          flag = False

      if flag:
        winning_contours.append(nominated_contours[i])

    for contour in nominated_contours:
      (x, y, w, h) = contour
      cv2.rectangle(previous_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for contour in winning_contours:
      (x, y, w, h) = contour
      cv2.rectangle(previous_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Video", previous_frame)
    previous_frame = current_frame
    _, current_frame = cap.read()

    k = cv2.waitKey(100)
    if k == 27 or k == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


fgbg = cv2.createBackgroundSubtractorMOG2(
  varThreshold=15,
  detectShadows=True
)

def get_mask(frame):
  mog_mask = fgbg.apply(frame)
  # median_blur_mask = cv2.medianBlur(mog_mask, 5)
  # bilateral_filter_mask = cv2.bilateralFilter(median_blur_mask, 9, 75, 75)
  # gaussian_blur_mask = cv2.GaussianBlur(bilateral_filter_mask, (13, 13), 5)

  # return gaussian_blur_mask
  return mog_mask


if __name__ == '__main__':
  display_video(r'..\datasets\RWF-2000\train\Fight\Ile3EVQA_0.avi')