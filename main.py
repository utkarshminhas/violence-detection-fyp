from preprocessing import novel_approach, video_to_numpy, video_to_frames
import paths


def main():
    novel_approach.display_preprocessed_video(video_path=paths._DATASET_PATH)
    # video_to_numpy.generate_numpy(paths.source_path, paths.target_path)
    # video_to_frames.generate_frames(source_path=paths.source_path, target_path=paths.target_path)


if __name__ == '__main__':
    main()
