import argparse
import sys

import cv2

from utils import apply_makeup, blur_background, create_face_mesh, create_segmenter, load_presets


def main(preset=None, blur_bg=False):
    presets = load_presets()
    names = list(presets)
    current = names.index(preset) if preset else 0

    # on windows the default (MSMF) backend can hang for a long time when opening the camera,
    # DirectShow opens it immediately. other platforms use the default backend.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    video_capture = cv2.VideoCapture(0, backend)
    if not video_capture.isOpened():
        sys.exit("Could not open camera.")

    # create the models once and reuse them for every frame, tracking mode for video
    with create_face_mesh(static_image_mode=False) as face_mesh, create_segmenter(video=True) as segmenter:
        while True:
            # read image from camera
            success, image = video_capture.read()
            if not success:
                break
            image = cv2.flip(image, 1)
            # blend the makeup into the frame, show the raw frame if no face is found
            output = apply_makeup(image, face_mesh, presets[names[current]])
            if output is None:
                output = image
            # blur everything except the person
            if blur_bg:
                output = blur_background(output, segmenter)
            cv2.imshow("Virtual Makeup", output)
            # q quits, b toggles the background blur, 1..9 switch preset
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                blur_bg = not blur_bg
            if ord("1") <= key <= ord("9") and key - ord("1") < len(names):
                current = key - ord("1")

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    preset_names = list(load_presets())
    parser = argparse.ArgumentParser(description="Webcam with facial makeup")
    parser.add_argument("--preset", choices=preset_names, default=preset_names[0], help="Makeup preset from presets.json.")
    parser.add_argument("--blur-background", action="store_true", help="Start with the background blurred (press b to toggle).")
    args = parser.parse_args()
    main(preset=args.preset, blur_bg=args.blur_background)
