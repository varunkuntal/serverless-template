import requests
import argparse
import os
import time
import numpy as np

import requests
from PIL import Image
from app import classify_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test server for ONNX model deployment')
    parser.add_argument('image_path', type=str, help='Path or URL to the image to classify')
    parser.add_argument('--api_url', type=str, default='http://banana-dev.example.com/predict', help='URL of the deployed ONNX model API')
    parser.add_argument('--test', action='store_true', help='Run preset tests')
    args = parser.parse_args()

    if args.test:
        images = [
            ('images/n01440764_tench.jpeg', 'tench'),
            ('images/n01667114_mud_turtle.JPEG', 'mud turtle'),
        ]
        for image_path, expected_class in images:
            class_name = classify_image(image_path)
            print(f'Image: {image_path}')
            print(f'Expected Class: {expected_class}')
            print(f'Classified As: {class_name}')
            print('-' * 50)

    start_time = time.time()
    class_name = classify_image(args.image_path)
    end_time = time.time()
    print(f'Classified image as: {class_name}')
    print(f'Time taken: {end_time - start_time:.4f}s')
