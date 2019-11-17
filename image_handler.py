"""
Loader class of images

Auther: Mehran Ahkami <ahkami.mehran@gmail.com>
"""
import os
import sys
import argparse
import cv2
import numpy as np

def load_image(file_path: str) -> np.ndarray:
    """Load image
    Arguments:
        file_path {str} -- path to file
    """
    try:
        img = cv2.imread(file_path)
    except Exception:
        print("error loading")
        sys.exit()
    return img


def save_image(image: np.ndarray, name:str ="data/noisy_image.jpg") -> None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())
    path = os.path.join(dir_path, name)
    if os.path.exists(os.path.join(dir_path, 'data')):
        try:
            cv2.imwrite(path, image)
        except Exception:
            print("There is an error in saving image")
            sys.exit()
    else:
        os.mkdir(os.path.join(dir_path, 'data'))
        try:
            cv2.imwrite(path, image)
        except Exception:
            print("There is an error in saving image")
            sys.exit()


def show_image(image:np.ndarray, window_time:int = 2500) -> None:
    cv2.imshow("image", image)
    cv2.waitKey(window_time)  # in ms
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("This is not correct format, give one path to image ")
        sys.exit()
    parser = argparse.ArgumentParser(description="Load image from a path.")
    parser.add_argument("path", metavar="N", type=str, nargs="+")
    parser.add_argument("--window", type=int,nargs="+", default=5000)
    args = parser.parse_args()
    image = load_image(file_path=args.path[0])
    show_image(image=image, window_time=args.window[0])
