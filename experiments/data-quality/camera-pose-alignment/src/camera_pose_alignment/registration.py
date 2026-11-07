"""Translation-only gradient ECC registration (method from M. Galletti, issue #89).

Sign convention: (dx, dy) is the shift of the scene content in the target image
relative to the reference, in pixels. Content moving right (dx > 0) means the
camera panned left.
"""

import cv2
import numpy as np

DEG_PER_PX = 54.2 / 1280  # horizontal FOV over image width, SDIS 77 cameras


def gradient(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = float(mag.max())
    return mag / m if m > 0 else mag


def prepare(gray: np.ndarray, crop_top: float) -> np.ndarray:
    return gradient(gray[int(gray.shape[0] * crop_top) :, :])


def register(ref_g: np.ndarray, img_g: np.ndarray, coarse: bool = True):
    """Return (dx, dy, ecc_correlation) or None if ECC fails.

    coarse=True seeds ECC with phase correlation so shifts larger than the
    ECC basin (a few px) are still found; coarse=False reproduces the original
    identity initialisation.
    """
    warp = np.eye(2, 3, dtype=np.float32)
    if coarse:
        win = cv2.createHanningWindow(ref_g.shape[::-1], cv2.CV_32F)
        (sx, sy), _ = cv2.phaseCorrelate(ref_g, img_g, win)
        warp[0, 2], warp[1, 2] = sx, sy
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(
            ref_g, img_g, warp, cv2.MOTION_TRANSLATION, criteria, None, 5
        )
    except cv2.error:
        return None
    return float(warp[0, 2]), float(warp[1, 2]), float(cc)


if __name__ == "__main__":
    # Self-check: a known sub-pixel shift on a textured image is recovered.
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((400, 1280)) * 255).astype(np.uint8), (0, 0), 2)
    for tx, ty in [(0.4, -0.3), (7.6, 2.2)]:
        m = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(base, m, base.shape[::-1], flags=cv2.INTER_CUBIC)
        dx, dy, cc = register(prepare(base, 0), prepare(shifted, 0))
        assert abs(dx - tx) < 0.05 and abs(dy - ty) < 0.05, (tx, ty, dx, dy)
    print("ok")
