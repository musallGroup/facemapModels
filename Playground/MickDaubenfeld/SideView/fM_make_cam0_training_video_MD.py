# make_cam0_training_video.py

from pathlib import Path
import cv2
import random

ROOT = Path(r"\\Naskampa\lts\BpodBehavior")

SESSIONS = [
    {
        "mouse": "427",
        "name": "PuffyPenguin",
        "session": "20251111_153254",
    },
    {
        "mouse": "462",
        "name": "UncertainUrchin",
        "session": "20260413_144048",
    },
    {
        "mouse": "460",
        "name": "PuffyPenguin",
        "session": "20260512_131654",
    },
]

OUTPUT_VIDEO = Path(r"\\Naskampa\lts\Team\Mick\Facemap_Models\Frontview_cam0\cam0_frontview_training_mix.avi")

CLIP_DURATION_SEC = 10
CLIPS_PER_VIDEO = 3
RANDOM_SEED = 42


def find_cam0_video(session_info):
    session_dir = (
        ROOT
        / session_info["mouse"]
        / session_info["name"]
        / "Session Data"
        / session_info["session"]
    )

    videos = list(session_dir.glob("*_cam0_*.avi"))

    if len(videos) == 0:
        raise FileNotFoundError(f"No cam0 video found in: {session_dir}")

    if len(videos) > 1:
        print(f"Multiple cam0 videos found in {session_dir}")
        for v in videos:
            print("  ", v)
        print("Using first one.")

    return videos[0]


def write_clip(cap, writer, start_frame, n_frames, target_size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written = 0
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if (frame.shape[1], frame.shape[0]) != target_size:
            frame = cv2.resize(frame, target_size)

        writer.write(frame)
        written += 1

    return written


def main():
    random.seed(RANDOM_SEED)

    video_paths = [find_cam0_video(s) for s in SESSIONS]

    print("Found videos:")
    for v in video_paths:
        print(v)

    first_cap = cv2.VideoCapture(str(video_paths[0]))
    if not first_cap.isOpened():
        raise RuntimeError(f"Could not open first video: {video_paths[0]}")

    fps = first_cap.get(cv2.CAP_PROP_FPS)
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_size = (width, height)

    first_cap.release()

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, target_size)

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Skipping, could not open: {video_path}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / video_fps

        print("\nProcessing:")
        print(video_path)
        print(f"FPS: {video_fps:.2f}, duration: {duration_sec:.1f} sec")

        clip_frames = int(CLIP_DURATION_SEC * video_fps)

        if total_frames <= clip_frames:
            start_frames = [0]
        else:
            max_start = total_frames - clip_frames
            start_frames = sorted(
                random.sample(
                    range(0, max_start),
                    k=min(CLIPS_PER_VIDEO, max_start)
                )
            )

        for start_frame in start_frames:
            written = write_clip(
                cap=cap,
                writer=writer,
                start_frame=start_frame,
                n_frames=clip_frames,
                target_size=target_size,
            )
            print(f"  wrote clip from frame {start_frame}, frames written: {written}")

        cap.release()

    writer.release()

    print("\nDone.")
    print(f"Output video saved here:\n{OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()