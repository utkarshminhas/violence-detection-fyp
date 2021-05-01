import cv2
import numpy as np
from functools import cmp_to_key
import os


class HoliApproachConfig:
  def __init__(self, image_size = (128, 128), padding_factor=0.1, max_take=2, contour_threshold=900, frame_break=5):
    self.image_size = image_size
    self.padding_factor = padding_factor
    self.max_take = max_take
    self.contour_threshold = contour_threshold
    self.frame_break = frame_break


class HoliApproach:
  def __init__(self, config):
    self.image_size = config.image_size
    self.padding_factor = config.padding_factor
    self.max_take = config.max_take
    self.contour_threshold = config.contour_threshold
    self.frame_break = config.frame_break


  def get_subvideo_arrays(self, src):
    cap = cv2.VideoCapture(src)

    _, previous_frame = cap.read()
    _, current_frame = cap.read()

    frame_count = 0

    whole_set = []
    boundaries = []
    current_set = []

    while cap.isOpened():
      if frame_count % self.frame_break == 1:
        mask = self.get_mask(previous_frame, current_frame)
        contours = self.get_contours(mask)
        nominated_contours = self.get_contour_positions(contours)
        boundaries = self.clean_contours(nominated_contours)

        if len(current_set) < self.max_take:
          whole_set.extend(current_set)
        else:
          whole_set.extend(current_set[:self.max_take])

        current_set = []
        for _ in boundaries:
          current_set.append([])

      for boundary_index, boundary in enumerate(boundaries):
        (x, y, w, h) = boundary

        cropped_frame = previous_frame[int(max(0, y)):int(min(cap.get(4), y + h)), int(max(0, x)):int(min(cap.get(3), x + w))]

        cropped_frame = cv2.resize(cropped_frame, self.image_size)
        current_set[boundary_index].append(cropped_frame)

      cv2.imshow("Activity", previous_frame)

      previous_frame = current_frame
      success, current_frame = cap.read()

      if not success or current_frame is None:
        break

      k = cv2.waitKey(1)
      if k == 27 or k == ord('q'):
        break

      frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return whole_set


  # def display_contours(self, video_path):
  #   cap = cv2.VideoCapture(video_path)
  # 
  #   _, previous_frame = cap.read()
  #   _, current_frame = cap.read()
  # 
  #   # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  #   # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  #   # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
  #   # out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))
  # 
  #   while cap.isOpened():
  #     mask = get_mask(previous_frame, current_frame)
  #     contours = get_contours(mask)
  #     nominated_contours = get_contour_positions(contours)
  #     winning_contours = clean_contours(nominated_contours)
  # 
  #     for contour in nominated_contours:
  #       (x, y, w, h) = contour
  #       cv2.rectangle(previous_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
  # 
  #     for contour in winning_contours:
  #       (x, y, w, h) = contour
  #       cv2.rectangle(previous_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  # 
  #     cv2.putText(previous_frame, f"{len(winning_contours)} + {len(nominated_contours) - len(winning_contours)} region(s)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
  # 
  #     # image = cv2.resize(previous_frame, (1280, 720))
  #     # out.write(image)
  # 
  #     cv2.imshow("Activity", previous_frame)
  #     previous_frame = current_frame
  #     _, current_frame = cap.read()
  # 
  #     k = cv2.waitKey(60)
  #     if k == 27 or k == ord('q'):
  #       break
  # 
  #   cap.release()
  #   cv2.destroyallwindows()
  #   # out.release()


  def generate_subvideos(self, src, tgt):
    whole_set = self.get_subvideo_arrays(src)

    for i, sub in enumerate(whole_set):
      fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

      if not os.path.exists(tgt):
        os.makedirs(tgt)

      out = cv2.VideoWriter(os.path.join(tgt, f"output_{str(i).zfill(4)}.avi"), fourcc, 5.0, self.image_size)

      for frame in sub:
        out.write(frame)

      out.release()

    return len(whole_set)


  def resize_contour(self, current_size):
    (x, y, w, h) = current_size

    if w / h > self.image_size[0] / self.image_size[1]:
      factor = w / self.image_size[0]
      delta = factor * self.image_size[1] - h
      y -= delta / 2
      h = factor * self.image_size[1]
    else:
      factor = h / self.image_size[1]
      delta = factor * self.image_size[0] - w
      x -= delta / 2
      w = factor * self.image_size[0]

    padding = min(h, w) * self.padding_factor

    x -= padding / 2
    y -= padding / 2
    w += padding
    h += padding

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    return (x, y, w, h)


  def clean_contours(self, contours):
    def is_contour_overlapping(i, j):
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


  def get_mask(self, previous_frame, current_frame):
    frame = cv2.absdiff(previous_frame, current_frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    _, frame = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
    frame = cv2.dilate(frame, None, iterations=3)

    return frame


  def get_contours(self, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


  def get_contour_positions(self, contours):
    nominated_contours = []

    for contour in contours:
      (x, y, w, h) = self.resize_contour(cv2.boundingRect(contour))

      if cv2.contourArea(contour) >= self.contour_threshold:
        nominated_contours.append((x, y, w, h))

    return nominated_contours


# if __name__ == '__main__':
#   filename = r'..\datasets\RWF-2000\train\Fight\0H2s9UJcNJ0_0.avi'
#   base_path = r'D:\Users\Madhavan\Repositories\Violence-Detection-FYP\preprocessing'
# 
#   config = HoliApproachConfig(
#     image_size=(128, 128),
#     padding_factor=0.1,
#     max_take=2,
#     contour_threshold=900,
#     frame_break=5
#   )
# 
#   holi_approach = HoliApproach(config)
# 
#   count = holi_approach.generate_subvideos(filename, base_path + "/prc")
#   print(count)
#   # display_contours(filename)
