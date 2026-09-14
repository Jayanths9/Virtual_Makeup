import argparse
import sys

import cv2

from utils import (
    FaceAnalyzer,
    adapt_style,
    blur_background,
    create_face_mesh,
    create_segmenter,
    detect_landmarks,
    load_presets,
    render_makeup,
    show_image,
    apply_foundation,
    hex_to_bgr,
)


def main(image_path, preset=None, blur_bg=False, adaptive=True, coverage=0.0, smoothing=0.0, shade=None):
    presets = load_presets()
    style = presets[preset] if preset else next(iter(presets.values()))
    # read image
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"Could not read image: {image_path}")
    # detect landmarks, adapt the shades to the person, smooth the skin, blend the makeup in
    with create_face_mesh(static_image_mode=True) as face_mesh:
        landmarks = detect_landmarks(image, face_mesh)
    if landmarks is None:
        print("No face detected, showing the original image.")
        output = image
    else:
        analysis = FaceAnalyzer().analyze(image, landmarks)
        if adaptive:
            style = adapt_style(style, analysis)
            print(f"Shades adapted for {analysis.describe()}")
        output = apply_foundation(image, landmarks, coverage, smoothing, shade, analysis)
        output = render_makeup(output, landmarks, style)
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
    parser.add_argument("--no-adapt", action="store_true", help="Use the preset shades as they are instead of adapting them to skin and light.")
    parser.add_argument("--foundation", type=float, default=0.0, metavar="0..1", help="Foundation coverage, evens out the skin colour (default 0 = off).")
    parser.add_argument("--smooth", type=float, default=0.0, metavar="0..1", help="Foundation texture smoothing (default 0).")
    parser.add_argument("--foundation-shade", metavar="#rrggbb", help="Foundation shade, matched to the skin when not given.")
    args = parser.parse_args()
    main(args.image, preset=args.preset, blur_bg=args.blur_background, adaptive=not args.no_adapt,
         coverage=args.foundation, smoothing=args.smooth,
         shade=hex_to_bgr(args.foundation_shade) if args.foundation_shade else None)
