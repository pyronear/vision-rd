import cv2
import numpy as np

from camera_pose_alignment.registration import prepare, register


def test_recovers_known_shift():
    rng = np.random.default_rng(0)
    noise = (rng.random((400, 1280)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(noise, (0, 0), 2)
    for tx, ty in [(0.4, -0.3), (7.6, 2.2)]:
        m = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(base, m, base.shape[::-1], flags=cv2.INTER_CUBIC)
        dx, dy, _ = register(prepare(base, 0), prepare(shifted, 0))
        assert abs(dx - tx) < 0.05 and abs(dy - ty) < 0.05
