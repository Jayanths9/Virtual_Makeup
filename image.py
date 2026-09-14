import argparse
import sys

import cv2

from utils import apply_makeup, blur_background, create_face_mesh, create_segmenter, load_presets, show_image


def main(image_path, preset=None, blur_bg=False):
    presets = load_presets()
    style = presets[preset] if preset else next(iter(presets.values()))
    # read image
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"Could not read image: {image_path}")
    # detect landmarks and blend the makeup into the image
    with create_face_mesh(static_image_mode=True) as face_mesh:
        output = apply_makeup(image, face_mesh, style)
    if output is None:
        print("No face detected, showing the original image.")
        output = image
    # blur everything except the person
    if blur_bg:
        with create_segmenter() as segmenter:
            output = blur_background(output, segmenter)
    # display the image
    show_image(output)


if __name__ == "__main__":
    preset_names = list(load_presets())
    parser = argparse.ArgumentParser(description="Add facial makeup to an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image.")
    parser.add_argument("--preset", choices=preset_names, default=preset_names[0], help="Makeup preset from presets.json.")
    parser.add_argument("--blur-background", action="store_true", help="Blur everything except the person.")
    args = parser.parse_args()
    main(args.image, preset=args.preset, blur_bg=args.blur_background)
