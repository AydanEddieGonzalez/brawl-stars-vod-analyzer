import cv2
import numpy as np
import pytest

from brawl_vod.frames import extract_frames, is_duplicate, iter_sampled_frames, thumbnail

FPS = 10
SIZE = (160, 120)  # width, height


def make_video(path, n_frames, changing=True):
    """Write a tiny test video. If changing, every frame looks different."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    rng = np.random.default_rng(0)
    still = rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
    for _ in range(n_frames):
        frame = rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8) if changing else still
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def moving_video(tmp_path):
    return make_video(tmp_path / "moving.avi", n_frames=50)  # 5 seconds


@pytest.fixture
def still_video(tmp_path):
    return make_video(tmp_path / "still.avi", n_frames=50, changing=False)


def test_thumbnail_shape_and_type():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    thumb = thumbnail(frame, size=64)
    assert thumb.shape == (64, 64)
    assert thumb.dtype == np.float32


def test_is_duplicate():
    a = np.zeros((64, 64), dtype=np.float32)
    assert is_duplicate(a, a + 1, threshold=4.0)
    assert not is_duplicate(a, a + 50, threshold=4.0)


def test_samples_one_frame_per_second(moving_video):
    times = [t for t, _ in iter_sampled_frames(moving_video, every_sec=1.0)]
    assert times == pytest.approx([0, 1, 2, 3, 4])


def test_skip_start_and_end(moving_video):
    times = [t for t, _ in iter_sampled_frames(moving_video, every_sec=1.0,
                                               skip_start=1.0, skip_end=1.0)]
    assert times == pytest.approx([1, 2, 3])


def test_extract_saves_changing_frames(moving_video, tmp_path):
    out = tmp_path / "frames"
    n = extract_frames(moving_video, out, skip_start=0, skip_end=0)
    assert n == 5
    assert len(list(out.glob("*.jpg"))) == 5


def test_extract_drops_duplicates(still_video, tmp_path):
    out = tmp_path / "frames"
    n = extract_frames(still_video, out, skip_start=0, skip_end=0)
    assert n == 1


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_sampled_frames(tmp_path / "nope.mp4"))
