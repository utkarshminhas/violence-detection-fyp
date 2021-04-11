import os
import shutil
import tqdm


def combine(base_path, all_videos_directory):
  tgt = os.path.join(base_path, all_videos_directory)

  if not os.path.exists(tgt):
    os.mkdir(tgt)

  all_converted = [f for f in os.listdir(base_path) if not os.path.isfile(os.path.join(base_path, f)) and f != all_videos_directory]

  counter = 1

  for video in tqdm.tqdm(all_converted):
    video_path = os.path.join(base_path, video)

    for subvideo in os.listdir(video_path):
      sub_tgt = os.path.join(tgt, f'video_{str(counter).zfill(5)}{os.path.splitext(subvideo)[1]}')
      shutil.move(os.path.join(video_path, subvideo), sub_tgt)
      counter += 1

    os.rmdir(video_path)


if __name__ == '__main__':
  combiner()
