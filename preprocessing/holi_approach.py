import cv2
import numpy as np
from functools import cmp_to_key
import os

scale_factor = 128

# video_001 -> [subvideo_01, subvideo_02, ...]
def holi_approach(video_path):
  subvideos = []

  cap = cv2.VideoCapture(video_path)

  _, previous_frame = cap.read()
  _, current_frame = cap.read()

  frame_count = 0
  boundaries = []

  whole_set = []
  current_set = []

  while cap.isOpened():
    if frame_count % 5 == 1:
      mask = get_mask(previous_frame, current_frame)
      contours = get_contours(mask)
      nominated_contours = get_contour_positions(contours)
      winning_contours = clean_contours(nominated_contours)

      boundaries = winning_contours

      whole_set.extend(current_set)

      current_set = []
      for _ in boundaries:
        current_set.append([])

    for boundary_index, boundary in enumerate(boundaries):
      (x, y, w, h) = boundary

      cropped_frame = previous_frame[int(max(0, y)):int(min(cap.get(4), y + h)), int(max(0, x)):int(min(cap.get(3), x + w))]

      cropped_frame = cv2.resize(cropped_frame, (128, 128))
      current_set[boundary_index].append(cropped_frame)

    cv2.imshow("Activity", previous_frame)

    previous_frame = current_frame
    success, current_frame = cap.read()

    if not success or current_frame is None:
      break

    k = cv2.waitKey(30)
    if k == 27 or k == ord('q'):
      break

    frame_count += 1

  cap.release()
  cv2.destroyAllWindows()

  for i, sub in enumerate(whole_set):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(f"output_{i}.avi", fourcc, 5.0, (128, 128))

    for frame in sub:
      out.write(frame)

    out.release()

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


def clean_contours(contours):
  def is_contour_overlapping(i, j): # Is i completely inside j?
      (ix, iy, iw, ih) = i
      (jx, jy, jw, jh) = j

      return ix > jx and ix + iw < jx + jw and iy > jy and iy + ih < jy + jh


  cleaned_contours = []
  contours = sorted(contours, key=cmp_to_key(lambda c1, c2: c2[2] * c2[3] - c1[2] * c1[3]))

  for i in range(len(contours)):
    for j in range(i):
      (ix, iy, iw, ih) = contours[i]
      (jx, jy, jw, jh) = contours[j]

      if is_contour_overlapping(contours[i], contours[j]):
        break
    else:
      cleaned_contours.append(contours[i])

  return cleaned_contours


def get_mask(previous_frame, current_frame):
  frame = cv2.absdiff(previous_frame, current_frame)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame = cv2.GaussianBlur(frame, (5,5), 0)
  _, frame = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
  frame = cv2.dilate(frame, None, iterations=3)

  return frame


def get_contours(mask):
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  return contours


def get_contour_positions(contours, threshold=900):
  nominated_contours = []

  for contour in contours:
    (x, y, w, h) = resize_contour(cv2.boundingRect(contour), aspect_ratio=(1, 1))

    if cv2.contourArea(contour) >= threshold:
      nominated_contours.append((x, y, w, h))

  return nominated_contours


def display_contours(video_path):
  cap = cv2.VideoCapture(video_path)

  _, previous_frame = cap.read()
  _, current_frame = cap.read()

  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
  out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

  while cap.isOpened():
    mask = get_mask(previous_frame, current_frame)
    contours = get_contours(mask)
    nominated_contours = get_contour_positions(contours)
    winning_contours = clean_contours(nominated_contours)

    for contour in nominated_contours:
      (x, y, w, h) = contour
      cv2.rectangle(previous_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for contour in winning_contours:
      (x, y, w, h) = contour
      cv2.rectangle(previous_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(previous_frame, f"{len(winning_contours)} + {len(nominated_contours) - len(winning_contours)} region(s)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    image = cv2.resize(previous_frame, (1280,720))
    out.write(image)

    cv2.imshow("Activity", previous_frame)
    previous_frame = current_frame
    _, current_frame = cap.read()

    k = cv2.waitKey(60)
    if k == 27 or k == ord('q'):
      break

  cap.release()
  cv2.destroyallwindows()
  out.release()


if __name__ == '__main__':
  holi_approach(r'..\datasets\RWF-2000\train\Fight\Ile3EVQA_0.avi')