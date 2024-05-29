import cv2
import argparse
from utils import *


# features to add makeup
face_elements = [
    "LIP_LOWER",
    "LIP_UPPER",
    "EYEBROW_LEFT",
    "EYEBROW_RIGHT",
    "EYELINER_LEFT",
    "EYELINER_RIGHT",
    "EYESHADOW_LEFT",
    "EYESHADOW_RIGHT",
]

# change the color of features
colors_map = {
    # upper lip and lower lips
    "LIP_UPPER": [0, 0, 255],  # Red in BGR
    "LIP_LOWER": [0, 0, 255],  # Red in BGR
    # eyeliner
    "EYELINER_LEFT": [139, 0, 0],  # Dark Blue in BGR
    "EYELINER_RIGHT": [139, 0, 0],  # Dark Blue in BGR
    # eye shadow
    "EYESHADOW_LEFT": [0, 100, 0],  # Dark Green in BGR
    "EYESHADOW_RIGHT": [0, 100, 0],  # Dark Green in BGR
    # eye brow
    "EYEBROW_LEFT": [19, 69, 139],  # Dark Brown in BGR
    "EYEBROW_RIGHT": [19, 69, 139],  # Dark Brown in BGR
}


def main(image_path):
    # extract required facial points from face_elements
    face_connections=[face_points[idx] for idx in face_elements]
    # extract corresponding colors for each facial features
    colors=[colors_map[idx] for idx in face_elements]
    # read image
    image = cv2.imread(image_path)
    # create a empty mask like image
    mask = np.zeros_like(image)
    # extract facial landmarks
    face_landmarks = read_landmarks(image=image)
    # create mask for facial features with color
    mask = add_mask(
        mask,
        idx_to_coordinates=face_landmarks,
    face_connections=face_connections,colors=colors
    )
    # combine the image and mask with w.r.to weights
    output = cv2.addWeighted(image, 1.0, mask, 0.2, 1.0)
    # display the image
    show_image(output)

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Image to add Facial makeup ")
    # add image path as argumment
    parser.add_argument("--img", type=str, help="Path to the image.")
    args = parser.parse_args()
    main(args.img)
