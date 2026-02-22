#!/usr/bin/env python3
"""Generate a synthetic face texture for the Gazebo face billboard.

Uses numpy + OpenCV (available in the sim container via ros-jazzy-cv-bridge).
Produces a face-like image designed to be detectable by the Haar frontal-face
cascade when rendered in Gazebo at ~60-80 px height from 2 m distance.

Run once before launching the sim (or after changing the face design):

    python3 sim/generate_face_texture.py

Output: sim/models/face_billboard/materials/textures/face_01.png
"""

import os

import cv2
import numpy as np


def _make_face(size: int = 512) -> np.ndarray:
    """Return a BGR uint8 face image of shape (size, size, 3).

    The face is designed for Haar frontal-face cascade detection:
    - High-contrast dark eye regions on a bright face region
    - Clear forehead area (lighter than cheeks)
    - Prominent mouth shadow below nose
    - Neutral gray background
    - Symmetric left-right (handles any UV-mirror from the box mesh)
    """
    # Neutral gray background
    img = np.full((size, size, 3), 160, dtype=np.uint8)

    cx = size // 2
    cy = int(size * 0.52)   # face centre slightly below image centre

    # ── Hair ──────────────────────────────────────────────────────────
    hair_colour = (35, 25, 15)
    cv2.ellipse(img, (cx, int(cy - size * 0.18)),
                (int(size * 0.38), int(size * 0.28)), 0, 0, 360, hair_colour, -1)

    # ── Face ellipse (skin tone) ───────────────────────────────────────
    face_colour = (165, 125, 90)        # moderate skin tone
    cv2.ellipse(img, (cx, cy),
                (int(size * 0.34), int(size * 0.42)), 0, 0, 360, face_colour, -1)

    # ── Forehead highlight (Haar needs bright-above-eyes region) ──────
    forehead_colour = (190, 155, 115)
    cv2.ellipse(img, (cx, int(cy - size * 0.18)),
                (int(size * 0.28), int(size * 0.15)), 0, 0, 360, forehead_colour, -1)

    # ── Eyebrow region (dark, above eyes — critical Haar feature) ─────
    brow_colour = (30, 20, 10)
    brow_y = int(cy - size * 0.10)
    eye_sep = int(size * 0.16)      # half-distance between eye centres
    brow_w = int(size * 0.12)
    brow_h = int(size * 0.035)
    for side in (-1, 1):
        cv2.ellipse(img, (cx + side * eye_sep, brow_y),
                    (brow_w, brow_h), 0, 0, 360, brow_colour, -1)
        # Blend brow edge back toward skin for natural look
        cv2.ellipse(img, (cx + side * eye_sep, brow_y + brow_h),
                    (brow_w, brow_h // 2), 0, 0, 360, face_colour, -1)

    # ── Eyes: white sclera + dark iris + black pupil ───────────────────
    eye_y = int(cy - size * 0.04)
    sclera_colour = (230, 228, 225)
    iris_colour   = (55, 38, 22)
    pupil_colour  = (8, 5, 5)
    hilight_colour = (255, 255, 255)
    sclera_rx = int(size * 0.11)
    sclera_ry = int(size * 0.075)
    iris_r    = int(size * 0.065)
    pupil_r   = int(size * 0.040)
    hi_r      = int(size * 0.018)
    hi_off    = int(size * 0.025)

    for side in (-1, 1):
        ex = cx + side * eye_sep
        cv2.ellipse(img, (ex, eye_y), (sclera_rx, sclera_ry),
                    0, 0, 360, sclera_colour, -1)
        cv2.circle(img, (ex, eye_y), iris_r, iris_colour, -1)
        cv2.circle(img, (ex, eye_y), pupil_r, pupil_colour, -1)
        cv2.circle(img, (ex + hi_off, eye_y - hi_off), hi_r, hilight_colour, -1)

    # ── Nose (subtle shadow strip — Haar uses nose-bridge region) ──────
    nose_top_y = eye_y + int(size * 0.07)
    nose_bot_y = int(cy + size * 0.12)
    nose_w     = int(size * 0.05)
    nose_colour = (140, 105, 75)
    cv2.ellipse(img, (cx, nose_bot_y),
                (nose_w, int(size * 0.03)), 0, 0, 360, nose_colour, -1)
    # Narrow dark bridge between eyes
    pts = np.array([
        [cx - nose_w // 2, nose_top_y],
        [cx + nose_w // 2, nose_top_y],
        [cx + nose_w,      nose_bot_y],
        [cx - nose_w,      nose_bot_y],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], nose_colour)

    # ── Mouth / chin (dark shadow — Haar uses lower-face gradient) ─────
    mouth_y  = int(cy + size * 0.20)
    mouth_rx = int(size * 0.14)
    mouth_ry = int(size * 0.035)
    lip_colour    = (110, 65, 60)
    shadow_colour = (125, 90, 65)

    # Lower lip area shadow
    cv2.ellipse(img, (cx, mouth_y + mouth_ry),
                (mouth_rx, mouth_ry + int(size * 0.02)), 0, 0, 180,
                shadow_colour, -1)
    # Upper lip line
    cv2.ellipse(img, (cx, mouth_y),
                (mouth_rx, mouth_ry), 0, 0, 180, lip_colour, 4)

    # ── Chin ──────────────────────────────────────────────────────────
    chin_y = int(cy + size * 0.38)
    cv2.ellipse(img, (cx, chin_y),
                (int(size * 0.18), int(size * 0.04)), 0, 0, 360,
                (145, 108, 78), -1)

    # ── Final blur to soften hard edges (improves cascade response) ────
    img = cv2.GaussianBlur(img, (9, 9), 3)

    return img


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    out_dir = os.path.join(
        repo_root, 'sim', 'models', 'face_billboard', 'materials', 'textures'
    )
    os.makedirs(out_dir, exist_ok=True)

    face = _make_face(size=512)
    out_path = os.path.join(out_dir, 'face_01.png')
    cv2.imwrite(out_path, face)
    print(f'Written {out_path}  ({face.shape[1]}×{face.shape[0]} px)')

    # Quick self-check: can Haar cascade detect it at representative scale?
    try:
        import glob
        import sys
        name = 'haarcascade_frontalface_default.xml'
        cascade_path = None
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            p = os.path.join(cv2.data.haarcascades, name)
            if os.path.exists(p):
                cascade_path = p
        if cascade_path is None:
            for d in sys.path:
                p = os.path.join(d, 'cv2', 'data', name)
                if os.path.exists(p):
                    cascade_path = p
                    break
        if cascade_path:
            cascade = cv2.CascadeClassifier(cascade_path)
            # Simulate ~76 px rendering at 2 m (640×480, 60° FOV)
            thumb = cv2.resize(face, (76, 76))
            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.05, 1, minSize=(20, 20))
            detected = len(faces) > 0
            print(f'Haar self-check at 76×76: {"DETECTED ✓" if detected else "not detected (expected at low res)"}')

            # Also check at full size
            gray_full = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces_full = cascade.detectMultiScale(gray_full, 1.1, 2, minSize=(80, 80))
            print(f'Haar self-check at 512×512: {"DETECTED ✓" if len(faces_full) > 0 else "not detected"}')
        else:
            print('Haar cascade XML not found — skipping self-check')
    except Exception as exc:
        print(f'Self-check skipped: {exc}')


if __name__ == '__main__':
    main()
