import cv2
from vanish_point import v_detect
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Synth')
    parser.add_argument('filename', help='path to image file [REQUIRED]')
    args = parser.parse_args()

    img = v_detect.load_image(args.filename)
    V = v_detect.detect(img, args.filename)

if __name__ == "__main__":
    main()