import contextlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

# the protobuf version mediapipe pulls in warns about its own deprecated call on every run
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh


def _chains(edges):
    """
    order an undirected edge set (like mediapipe's FACEMESH_* constants) into point chains.
    returns [(points, closed), ...] where closed is True for a loop
    """
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    chains, seen = [], set()
    # open chains must start at an end point (one neighbour), loops can start anywhere
    for start in sorted(adjacency, key=lambda p: (len(adjacency[p]) != 1, p)):
        if start in seen:
            continue
        chain, current = [start], start
        seen.add(start)
        while True:
            unvisited = sorted(n for n in adjacency[current] if n not in seen)
            if not unvisited:
                break
            current = unvisited[0]
            chain.append(current)
            seen.add(current)
        chains.append((chain, start in adjacency[current] and len(chain) > 2))
    return chains


_MESH_NEIGHBOURS = {}
for _a, _b in mp_face_mesh.FACEMESH_TESSELATION:
    _MESH_NEIGHBOURS.setdefault(_a, set()).add(_b)
    _MESH_NEIGHBOURS.setdefault(_b, set()).add(_a)


def _loop(edges, containing=None):
    """the closed loop in an edge set, or the one containing a given point if there are several"""
    loops = [chain for chain, closed in _chains(edges) if closed]
    if containing is not None:
        loops = [chain for chain in loops if containing in chain]
    assert len(loops) == 1, "expected exactly one loop"
    return loops[0]


def _band(edges):
    """
    close the two parallel open chains of an edge set (the top and bottom edge of an eyebrow)
    into one polygon. if both chains run the same way their last points are mesh neighbours.
    """
    (a, _), (b, _) = _chains(edges)
    return a + b[::-1] if b[-1] in _MESH_NEIGHBOURS[a[-1]] else a + b


# landmark indices of each facial feature polygon (mediapipe face mesh, 0..467).
# eyes, eyebrows, lips and the face oval are derived from the contours mediapipe ships, so they
# cannot drift from the library. mediapipe names sides from the subjects point of view, this
# project names them as seen in the image, hence the swap.
# eyeshadow and eyeliner have no canonical contour and are picked by hand.
# BLUSH_* and FACE are not used yet, kept for future features.
face_points = {
    "LEFT_EYE": _loop(mp_face_mesh.FACEMESH_RIGHT_EYE),
    "RIGHT_EYE": _loop(mp_face_mesh.FACEMESH_LEFT_EYE),
    "EYEBROW_LEFT": _band(mp_face_mesh.FACEMESH_RIGHT_EYEBROW),
    "EYEBROW_RIGHT": _band(mp_face_mesh.FACEMESH_LEFT_EYEBROW),
    "LIPS_OUTER": _loop(mp_face_mesh.FACEMESH_LIPS, containing=0),
    "LIPS_INNER": _loop(mp_face_mesh.FACEMESH_LIPS, containing=13),
    "FACE": _loop(mp_face_mesh.FACEMESH_FACE_OVAL),
    "EYESHADOW_LEFT": [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130],
    "EYESHADOW_RIGHT": [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362],
    "EYELINER_LEFT": [243, 112, 26, 22, 23, 24, 110, 25, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133],
    "EYELINER_RIGHT": [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 359, 446, 255, 339, 254, 253, 252, 256, 341],
    "BLUSH_LEFT": [50],
    "BLUSH_RIGHT": [280],
}
assert all(0 <= i < 468 for points in face_points.values() for i in points), "landmark index out of range"

# the features a user can control, in painting order. each one paints a set of polygons and
# cuts out its holes (the mouth opening for the lips).
# "lightness" is how strongly the lightness of the skin follows the chosen colour:
# 0 keeps the skin shading and texture fully, 1 paints a flat colour.
FEATURES = {
    "lips": {"label": "Lips", "polygons": ["LIPS_OUTER"], "holes": ["LIPS_INNER"], "lightness": 0.25},
    "eyeshadow": {"label": "Eyeshadow", "polygons": ["EYESHADOW_LEFT", "EYESHADOW_RIGHT"], "holes": [], "lightness": 0.10},
    "eyeliner": {"label": "Eyeliner", "polygons": ["EYELINER_LEFT", "EYELINER_RIGHT"], "holes": [], "lightness": 0.60},
    "eyebrows": {"label": "Eyebrows", "polygons": ["EYEBROW_LEFT", "EYEBROW_RIGHT"], "holes": [], "lightness": 0.30},
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


# webcam modes by name, smallest first
CAMERA_MODES = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}
CAMERA_RESOLUTION_CHOICES = ["auto", "detect", "max"] + list(CAMERA_MODES)
# "auto" stops climbing here: above it the makeup pipeline itself gets slow
AUTO_MAX_MODE = "1080p"
# a mode has to keep at least this frame rate to count as usable for live video
MIN_SMOOTH_FPS = 20.0
# where the result of an "auto" probe is remembered per webcam
CAMERA_CACHE = Path.home() / ".virtual_makeup" / "cameras.json"


def _read_camera_cache() -> dict:
    try:
        with open(CAMERA_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def camera_is_known(index: int) -> bool:
    """True if an "auto" probe of this webcam was already done and remembered"""
    return str(index) in _read_camera_cache()


def _write_camera_cache(cache: dict):
    try:
        CAMERA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CAMERA_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass  # remembering is a convenience, never a failure


def _open_capture(index: int, size: tuple | None = None, mjpg: bool = False):
    """open a webcam, optionally straight into a frame size and MJPG. None if it cannot be opened"""
    # on windows the default (MSMF) backend can hang for a long time when opening the camera,
    # DirectShow opens it immediately. other platforms use the default backend.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    params = []
    if size is not None:
        params += [cv2.CAP_PROP_FRAME_WIDTH, size[0], cv2.CAP_PROP_FRAME_HEIGHT, size[1]]
    if mjpg:
        params += [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")]
    capture = cv2.VideoCapture(index, backend, params)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


class _Camera:
    """
    a VideoCapture plus what it is currently set to. every property change is a multi second
    stream rebuild on some drivers, so requests that change nothing are skipped.
    """

    def __init__(self, index: int, size: tuple | None = None, mjpg: bool = False):
        self.index = index
        self.capture = _open_capture(index, size, mjpg)
        self.size = None
        self.mjpg = mjpg
        if self.capture is not None:
            # the first frame after opening takes up to a second, get it out of the way so
            # frame rate measurements are not skewed
            ok, frame = self.capture.read()
            self.size = (frame.shape[1], frame.shape[0]) if ok else None

    def set_mode(self, size: tuple, mjpg: bool):
        """request a frame size (and MJPG), return the size the camera actually delivers or None"""
        if size != self.size:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            # a size change resets the format to the native one, MJPG must be requested after it
            self.mjpg = False
        if mjpg and not self.mjpg:
            # cameras without MJPG ignore this and keep their native format
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.mjpg = True
        # drivers report a requested size without honouring it, only a real frame is proof
        ok, frame = self.capture.read()
        self.size = (frame.shape[1], frame.shape[0]) if ok else None
        return self.size

    def fps(self, frames: int = 8, budget: float = 0.5) -> float:
        """frames per second the capture delivers right now, 0 if it stops delivering"""
        start, count = time.perf_counter(), 0
        while count < frames and time.perf_counter() - start < budget:
            ok, _ = self.capture.read()
            if not ok:
                return 0.0
            count += 1
        return count / max(time.perf_counter() - start, 1e-6)

    def reopen(self):
        """a fresh capture in its default mode, the only way back from MJPG to the native format"""
        self.capture.release()
        self.__init__(self.index)


def _probe(camera: _Camera, names: list, need_fps: bool):
    """
    climb through the given modes and return (size, mjpg) of the best usable one, or None.
    with need_fps a mode must keep MIN_SMOOTH_FPS, first natively, then as MJPG.
    """
    best, tried_mjpg = None, False
    for name in names:
        size = CAMERA_MODES[name]
        delivered = camera.set_mode(size, camera.mjpg)
        if delivered is None or delivered[0] < size[0]:
            # the camera cannot do this size, it will not do anything bigger either
            break
        if not need_fps or camera.fps() >= MIN_SMOOTH_FPS:
            best = (size, camera.mjpg)
            continue
        # too slow uncompressed, a compressed stream may keep the frame rate up. one try:
        # a camera that ignores the request the first time will ignore it every time
        if not tried_mjpg:
            tried_mjpg = True
            delivered = camera.set_mode(size, True)
            if delivered and delivered[0] >= size[0] and camera.fps() >= MIN_SMOOTH_FPS:
                best = (size, True)
                continue
        break
    return best


def remember_camera_mode(index: int, size: tuple, mjpg: bool):
    """make "auto" open this webcam in the given mode from now on"""
    cache = _read_camera_cache()
    cache[str(index)] = {"size": [int(size[0]), int(size[1])], "mjpg": bool(mjpg)}
    _write_camera_cache(cache)


def probe_camera(index: int = 0, progress=None) -> list:
    """
    measure every mode of a webcam, native format first and MJPG where the native one is too
    slow. takes a while: each mode switch is a multi second stream rebuild on some drivers.

    index : webcam number
    progress : optional callable taking a short status string, called before every measurement
    returns rows in the order tested, each
        {"mode": "720p", "size": (w, h), "format": "MJPG", "fps": 30.5, "mjpg": True,
         "smooth": True, "recommended": False}
    the recommended row is the largest smooth mode up to AUTO_MAX_MODE, it is also remembered
    as the "auto" choice. an empty list means the webcam could not be opened.
    """
    camera = _Camera(index)
    if camera.capture is None:
        return []
    rows = []
    for name, size in CAMERA_MODES.items():
        for mjpg in (False, True):
            if progress:
                progress(f"Testing {name} {'MJPG' if mjpg else 'native'}...")
            delivered = camera.set_mode(size, mjpg)
            if delivered is None or delivered[0] < size[0]:
                # the camera cannot do this size, it will not do anything bigger either
                break
            fps = camera.fps(frames=15, budget=1.5)
            fmt = camera_format(camera.capture)
            rows.append({"mode": name, "size": delivered, "format": fmt, "fps": fps, "mjpg": mjpg,
                         "smooth": fps >= MIN_SMOOTH_FPS, "recommended": False})
            if fps >= MIN_SMOOTH_FPS or fmt == "MJPG":
                # smooth already, or MJPG is what we just tried: no point testing the other format
                break
        else:
            continue
        if delivered is None or delivered[0] < size[0]:
            break
    camera.capture.release()

    usable = [r for r in rows if r["smooth"] and list(CAMERA_MODES).index(r["mode"]) <= list(CAMERA_MODES).index(AUTO_MAX_MODE)]
    if usable:
        best = max(usable, key=lambda r: r["size"][0] * r["size"][1])
        best["recommended"] = True
        remember_camera_mode(index, best["size"], best["mjpg"])
    return rows


def open_camera(index: int = 0, resolution: str = "auto"):
    """
    index : webcam number
    resolution : "auto"   - the largest mode up to AUTO_MAX_MODE that still runs smoothly.
                            bigger modes on a webcam are often upscaled by the driver and run at
                            a fraction of the frame rate, so this is measured, not assumed. the
                            result is remembered in CAMERA_CACHE, later starts skip the probing
                 "detect" - like auto, but measures again and updates the cache
                 "max"    - the largest size the camera delivers, whatever the frame rate
                 a name from CAMERA_MODES ("720p") - that mode
    returns an opened cv2.VideoCapture, or None if the camera cannot be opened
    """
    cache = _read_camera_cache()
    if resolution == "auto" and str(index) in cache:
        # open straight into the remembered mode, every later property change costs seconds
        remembered = cache[str(index)]
        camera = _Camera(index, tuple(remembered["size"]), remembered["mjpg"])
        return camera.capture
    if resolution in CAMERA_MODES:
        camera = _Camera(index, CAMERA_MODES[resolution])
        if camera.capture is not None and camera.fps() < MIN_SMOOTH_FPS:
            camera.set_mode(CAMERA_MODES[resolution], True)
        return camera.capture

    camera = _Camera(index)
    if camera.capture is None:
        return None
    if resolution in ("auto", "detect", "max"):
        names = list(CAMERA_MODES)
        if resolution != "max":
            names = names[: names.index(AUTO_MAX_MODE) + 1]
        best = _probe(camera, names, need_fps=resolution != "max")
        if best is None:
            # not even the smallest mode works smoothly, keep whatever the camera is in now
            return camera.capture
        if resolution != "max":
            cache[str(index)] = {"size": list(best[0]), "mjpg": best[1]}
            _write_camera_cache(cache)
        if camera.mjpg and not best[1]:
            camera.reopen()
            if camera.capture is None:
                return None
        camera.set_mode(best[0], best[1])
        return camera.capture

    raise ValueError(f"unknown resolution {resolution!r}, expected one of {CAMERA_RESOLUTION_CHOICES}")


def camera_resolution(capture) -> tuple:
    """(width, height) the capture is delivering"""
    return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def camera_format(capture) -> str:
    """four character code of the frame format the capture is delivering, e.g. YUY2 or MJPG"""
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    name = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip()
    # real codes are short ascii names like YUY2, MJPG or NV12, anything else is garbage
    return name if name.isascii() and name.isalnum() else "?"


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
        if spec["holes"]:
            cv2.fillPoly(mask, [landmarks[face_points[h]] - (x0, y0) for h in spec["holes"]], 0.0)
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
    # the model works on a small input anyway, feed it a downscaled frame to save the conversion
    small = cv2.resize(image, (w // 4, h // 4), interpolation=cv2.INTER_AREA) if min(h, w) >= 480 else image
    # per pixel probability of being the person, 0..1, mediapipe expects RGB
    mask = segmenter.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).segmentation_mask
    # tighten the transition, then feather it so the cut-out edge is not jagged
    mask = np.clip((mask - 0.3) / 0.4, 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(mask.shape) / 200))
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    # a blur is low frequency, so blur a quarter-size copy and scale it back up: same look,
    # a fraction of the cost at high resolutions. kernel scales with size so the look is stable.
    kernel = max(3, int(min(small.shape[:2]) * strength) | 1)
    blurred = cv2.resize(cv2.stackBlur(small, (kernel, kernel)), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.blendLinear(image, blurred, mask, 1 - mask)
