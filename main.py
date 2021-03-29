from preprocessing import novel_approach, video_to_numpy, video_to_frames, motion_detection
import paths


def main():
    # novel_approach.display_preprocessed_video(video_path=paths._DATASET_PATH)
    # video_to_numpy.generate_numpy(paths.source_path, paths.target_path)
    # video_to_frames.generate_frames(source_path=paths.source_path, target_path=paths.target_path)
    motion_detection.motion_detection(video_path=paths.SOME_OTHER_FILE)


if __name__ == '__main__':
    main()
