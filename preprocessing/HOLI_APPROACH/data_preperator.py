import subvideo_generator as svg
import combiner as cbr
from holi_approach import HoliApproachConfig

base_path = r'D:\Users\Madhavan\Repositories\Violence-Detection-FYP\datasets\RWF-2000\train\Fight'

config = HoliApproachConfig()
config.image_size=(128, 128)
config.padding_factor=0.1
config.max_take=5
config.contour_threshold=900
config.frame_break=5

all_videos_path = 'all_videos'


def main():
  print('Base Path: {base_path}')

  print()

  print('Generating subvideos...')
  svg.generate_subvideos(base_path, config)
  print('Done')

  print()

  print('Combining results...')
  cbr.combine(base_path, all_videos_path)
  print('Done')


if __name__ == '__main__':
    main()
