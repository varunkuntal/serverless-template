import requests
import argparse
import os
import time
import numpy as np
from urllib.parse import urlparse
from io import BytesIO

import requests
from PIL import Image
from model import OnnxModel, Preprocessor

preprocessor = Preprocessor()
onnx_model = OnnxModel("model/model_optimized.onnx")

def get_class_names():
    filename = "imagenet_class_names.txt"
    try:
        with open(filename) as f:
            class_names = f.readlines()
        return [c.strip() for c in class_names]
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = requests.get(url)
        class_names = response.json()
        with open(filename, "w") as f:
            for c in class_names:
                f.write(c + "\n")
        return class_names[:1000]


def classify_image(image_path, onnx_model=onnx_model, preprocessor=preprocessor):
    if not isinstance(image_path, str):
        raise ValueError("Image path must be a string")

    if urlparse(image_path).scheme in ['http', 'https']:
        # Download the file if image_path is a URL
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        # Use local file path if image_path is not a URL
        if not os.path.isfile(image_path):
            raise ValueError(f"Invalid file path: {image_path}")
        img = Image.open(image_path)

    img_processed = preprocessor(img)
    output = onnx_model.predict(img_processed)
    predicted_classes = np.argmax(output[0], axis=1)
    class_name = get_class_names()[predicted_classes[0]]
    return class_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test server for ONNX model deployment')
    parser.add_argument('image_path', type=str, help='Path to the image to classify')
    parser.add_argument('--test', action='store_true', help='Run preset tests')
    args = parser.parse_args()

    onnx_model = OnnxModel(os.path.join('model', 'model_optimized.onnx'))
    preprocessor = Preprocessor()

    if args.test:
        images = [
            ('images/n01440764_tench.jpeg', 'tench'),
            ('images/n01667114_mud_turtle.JPEG', 'mud turtle'),
        ]
        for image_path, expected_class in images:
            class_name = classify_image(image_path, onnx_model, preprocessor)
            print(f'Image: {image_path}')
            print(f'Expected Class: {expected_class}')
            print(f'Classified As: {class_name}')
            print('-' * 50)

    start_time = time.time()
    class_name = classify_image(args.image_path, onnx_model, preprocessor)
    end_time = time.time()
    print(f'Classified image as: {class_name}')
    print(f'Time taken: {end_time - start_time:.4f}s')
