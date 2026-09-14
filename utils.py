import contextlib
import json
import os
import sys
import warnings
from pathlib import Path

# the protobuf version mediapipe pulls in warns about its own deprecated call on every run
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import cv2
import mediapipe as mp
import numpy as np


# landmark indices of each facial feature polygon (mediapipe face mesh, 0..467)
# BLUSH_*, LEFT_EYE, RIGHT_EYE and FACE are not used yet, kept for future features
face_points = {
    "BLUSH_LEFT": [50],
    "BLUSH_RIGHT": [280],
    "LEFT_EYE": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "RIGHT_EYE": [362, 298, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    "EYELINER_LEFT": [243, 112, 26, 22, 23, 24, 110, 25, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133],
    "EYELINER_RIGHT": [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 359, 446, 255, 339, 254, 253, 252, 256, 341],
    "EYESHADOW_LEFT": [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130],
    "EYESHADOW_RIGHT": [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362],
    "FACE": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 454, 323, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377],
    "LIP_UPPER": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78],
    "LIP_LOWER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78],
    "EYEBROW_LEFT": [55, 107, 66, 105, 63, 70, 46, 53, 52, 65],
    "EYEBROW_RIGHT": [285, 336, 296, 334, 293, 300, 276, 283, 295],
}

# the features a user can control, in painting order. each one paints a set of polygons.
# "lightness" is how strongly the lightness of the skin follows the chosen colour:
# 0 keeps the skin shading and texture fully, 1 paints a flat colour.
FEATURES = {
    "lips": {"label": "Lips", "polygons": ["LIP_UPPER", "LIP_LOWER"], "lightness": 0.25},
    "eyeshadow": {"label": "Eyeshadow", "polygons": ["EYESHADOW_LEFT", "EYESHADOW_RIGHT"], "lightness": 0.10},
    "eyeliner": {"label": "Eyeliner", "polygons": ["EYELINER_LEFT", "EYELINER_RIGHT"], "lightness": 0.60},
    "eyebrows": {"label": "Eyebrows", "polygons": ["EYEBROW_LEFT", "EYEBROW_RIGHT"], "lightness": 0.30},
}

PRESETS_FILE = Path(__file__).with_name("presets.json")


def hex_to_bgr(value: str) -> tuple:
    """'#rrggbb' -> (b, g, r)"""
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def bgr_to_hex(color) -> str:
    """(b, g, r) -> '#rrggbb'"""
    b, g, r = (int(c) for c in color)
    return "#%02x%02x%02x" % (r, g, b)


def load_presets(path=PRESETS_FILE) -> dict:
    """
    path : json file of {preset name: {feature: {"color": "#rrggbb", "alpha": 0..1, "enabled": bool}}}
    returns {preset name: style}. a style has an entry for every feature in FEATURES:
    {feature: {"color": (b, g, r), "alpha": 0..1, "enabled": bool}}
    features missing from a preset are disabled.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    presets = {}
    for name, spec in raw.items():
        style = {}
        for feature in FEATURES:
            entry = spec.get(feature)
            if entry is None:
                style[feature] = {"color": (0, 0, 0), "alpha": 0.0, "enabled": False}
                continue
            style[feature] = {
                "color": hex_to_bgr(entry["color"]),
                "alpha": float(entry.get("alpha", 0.5)),
                "enabled": bool(entry.get("enabled", True)),
            }
        presets[name] = style
    return presets


# initialize mediapipe functions
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# tiny frame used to run each model once right after it is created
_WARMUP_FRAME = np.zeros((64, 64, 3), np.uint8)


@contextlib.contextmanager
def _quiet_stderr():
    """
    mediapipe's C++ layer prints INFO / WARNING lines straight to the process stderr while a
    model is created and on its first inference, bypassing python logging entirely.
    temporarily point file descriptor 2 at devnull for that window.
    """
    try:
        saved = os.dup(2)
    except OSError:
        # no usable stderr (for example pythonw), nothing to silence
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        if sys.stderr:
            sys.stderr.flush()
        os.dup2(devnull, 2)
        yield
    finally:
        if sys.stderr:
            sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def create_face_mesh(static_image_mode: bool = True):
    """
    static_image_mode : True for single images, False for video
                        (False enables tracking between frames, which is faster and less jittery)
    returns a FaceMesh instance, use it as a context manager and create it only once
    """
    with _quiet_stderr():
        # refine_landmarks would only add the iris points 468..477, which no feature uses
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=False,
        )
        # the first inference loads the model and logs, do it here so real frames are quiet and fast
        face_mesh.process(_WARMUP_FRAME)
    return face_mesh


def create_segmenter(video: bool = False):
    """
    video : True selects the lighter model meant for webcam frames
    returns a SelfieSegmentation instance, use it as a context manager and create it only once
    """
    with _quiet_stderr():
        segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1 if video else 0)
        segmenter.process(_WARMUP_FRAME)
    return segmenter


# to display image in cv2 window
def show_image(image: np.ndarray, msg: str = "Virtual Makeup"):
    """
    image : image as np array
    msg : cv2 window name
    """
    cv2.imshow(msg, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_landmarks(image: np.ndarray, face_mesh):
    """
    image : BGR image as np.ndarray
    face_mesh : initialized FaceMesh instance (see create_face_mesh)
    returns an (468, 2) int32 array with the pixel (x, y) of every landmark, or None if no face
    is found. points may fall outside the image, cv2.fillPoly clips them.
    """
    # mediapipe expects RGB input, opencv reads images as BGR
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    h, w = image.shape[:2]
    # landmarks come normalized to 0..1, scale them to the image size
    normalized = np.array([(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark], np.float32)
    return np.rint(normalized * (w, h)).astype(np.int32)


def feather_size(landmarks: np.ndarray) -> int:
    """odd blur kernel size scaled to the face, from the distance between the outer eye corners"""
    eye_distance = float(np.linalg.norm(landmarks[33] - landmarks[263]))
    return int(max(3, round(eye_distance * 0.06))) // 2 * 2 + 1


def _bgr_to_lab(color) -> np.ndarray:
    pixel = np.array([[color]], np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def render_makeup(image: np.ndarray, landmarks: np.ndarray, style: dict) -> np.ndarray:
    """
    image : BGR image as np.ndarray
    landmarks : array from detect_landmarks
    style : {feature: {"color": (b, g, r), "alpha": 0..1, "enabled": bool}} (see load_presets)
    returns a new BGR image with the makeup blended in.

    blending happens in LAB colour space: the a/b (colour) channels move toward the chosen
    colour while L (lightness) mostly keeps the skin's shading and texture, so a colour looks the
    same on any skin tone and lips keep their highlights. each feature only touches the pixels
    around its own polygons.
    """
    h, w = image.shape[:2]
    output = image.copy()
    kernel = feather_size(landmarks)
    for name, spec in FEATURES.items():
        entry = style.get(name)
        if not entry or not entry["enabled"] or entry["alpha"] <= 0:
            continue
        polygons = [landmarks[face_points[p]] for p in spec["polygons"]]
        # work on the bounding box of the polygons plus the blur reach
        points = np.concatenate(polygons)
        x0, y0 = np.maximum(points.min(axis=0) - kernel, 0)
        x1, y1 = np.minimum(points.max(axis=0) + kernel + 1, (w, h))
        if x1 <= x0 or y1 <= y0:
            # feature entirely outside the image
            continue
        # soft coverage mask of the feature, 0..alpha
        mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.fillPoly(mask, [p - (x0, y0) for p in polygons], 1.0)
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0) * entry["alpha"]
        # blend in LAB inside the box
        target = _bgr_to_lab(entry["color"])
        region = output[y0:y1, x0:x1]
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        m = mask[..., None]
        lightness = m * spec["lightness"]
        lab[..., :1] = lab[..., :1] * (1 - lightness) + target[0] * lightness
        lab[..., 1:] = lab[..., 1:] * (1 - m) + target[1:] * m
        blended = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        # only write pixels the mask actually covers, the colour space round trip is lossy
        covered = mask > 0
        region[covered] = blended[covered]
    return output


def apply_makeup(image: np.ndarray, face_mesh, style: dict):
    """
    image : BGR image as np.ndarray
    face_mesh : initialized FaceMesh instance (see create_face_mesh)
    style : {feature: {"color": (b, g, r), "alpha": 0..1, "enabled": bool}} (see load_presets)
    returns the image with makeup blended in, or None if no face is found
    """
    landmarks = detect_landmarks(image, face_mesh)
    if landmarks is None:
        return None
    return render_makeup(image, landmarks, style)


def blur_background(image: np.ndarray, segmenter, strength: float = 0.05):
    """
    image : BGR image as np.ndarray
    segmenter : initialized SelfieSegmentation instance (see create_segmenter)
    strength : blur kernel size as a fraction of the shorter image side
    returns the image with everything except the person blurred
    """
    h, w = image.shape[:2]
    # per pixel probability of being the person, 0..1, mediapipe expects RGB
    mask = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).segmentation_mask
    # tighten the transition, then feather it so the cut-out edge is not jagged
    mask = np.clip((mask - 0.3) / 0.4, 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(h, w) / 200))[..., None]
    # kernel scales with image size so the effect looks the same at any resolution
    kernel = max(3, int(min(h, w) * strength) | 1)
    blurred = cv2.stackBlur(image, (kernel, kernel))
    return (image * mask + blurred * (1 - mask)).astype(np.uint8)
