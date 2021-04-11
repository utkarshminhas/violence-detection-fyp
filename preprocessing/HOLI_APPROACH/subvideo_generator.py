import os
from holi_approach import HoliApproachConfig, HoliApproach
import tqdm


def generate_subvideos(base_path, config):
  holi_approach = HoliApproach(config)

  all_videos = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

  for video in tqdm.tqdm(all_videos):
    video_base_name = os.path.splitext(video)[0]

    src = os.path.join(base_path, video)
    tgt = os.path.join(base_path, video_base_name)

    if os.path.exists(tgt):
        continue

    count = holi_approach.generate_subvideos(src, tgt)


if __name__ == '__main__':
  subvideo_generator()
