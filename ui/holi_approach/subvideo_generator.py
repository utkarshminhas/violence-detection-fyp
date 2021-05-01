import os
from holi_approach.holi_approach import HoliApproachConfig, HoliApproach
import tqdm


def generate_video(video_path, config):
  holi_approach = HoliApproach(config)

  src = video_path
  tgt = os.path.join('tmp', 'current')

  count = holi_approach.generate_subvideos(src, tgt)

  return count
