import argparse
import sys

import cv2

from utils import (
    CAMERA_RESOLUTION_CHOICES,
    apply_makeup,
    blur_background,
    camera_format,
    camera_resolution,
    create_face_mesh,
    create_segmenter,
    load_presets,
    open_camera,
    probe_camera,
)

WINDOW = "Virtual Makeup"
DISPLAY_MAX_WIDTH = 1280


def test_camera(index=0):
    """measure every webcam mode and print the table, the recommended one becomes the auto choice"""
    print("Testing webcam modes, each switch takes a few seconds...")
    rows = probe_camera(index, progress=lambda text: print("  " + text))
    if not rows:
        sys.exit("Could not open camera.")
    print()
    print("  %-7s %-11s %-7s %-9s %s" % ("mode", "delivered", "format", "fps", "verdict"))
    for row in rows:
        verdict = "recommended, used by auto" if row["recommended"] else "smooth" if row["smooth"] else "too slow"
        print("  %-7s %-11s %-7s %5.1f     %s" % (row["mode"], "%dx%d" % row["size"], row["format"], row["fps"], verdict))


def main(preset=None, blur_bg=False, resolution="auto"):
    presets = load_presets()
    names = list(presets)
    current = names.index(preset) if preset else 0

    if resolution == "auto":
        print("Detecting the best webcam mode...")
    video_capture = open_camera(0, resolution)
    if video_capture is None:
        sys.exit("Could not open camera.")
    width, height = camera_resolution(video_capture)
    print(f"Camera: {width}x{height} {camera_format(video_capture)}")
    # frames wider than the display size are scaled down properly before showing
    display_width = min(width, DISPLAY_MAX_WIDTH)

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
            if output.shape[1] > display_width:
                scale = display_width / output.shape[1]
                output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, output)
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
    parser.add_argument("--resolution", choices=CAMERA_RESOLUTION_CHOICES, default="auto",
                        help="Webcam mode: auto picks the largest that still runs smoothly (default).")
    parser.add_argument("--test", action="store_true", help="Measure every webcam mode, print the results and exit.")
    args = parser.parse_args()
    if args.test:
        test_camera()
    else:
        main(preset=args.preset, blur_bg=args.blur_background, resolution=args.resolution)
