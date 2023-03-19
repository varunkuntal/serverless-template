import onnxruntime
from PIL import Image
from model import Preprocessor, OnnxModel
import numpy as np
from urllib.parse import urlparse
from io import BytesIO
import requests
import os

model = OnnxModel('model/model_optimized.onnx')
preprocessor = Preprocessor()

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global preprocessor, model
    
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

def classify_image(image_path, onnx_model=model, preprocessor=preprocessor):
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

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global preprocessor, model

    # Parse out your arguments
    img_url = model_inputs.get('url', None)

    if img_url is None:
        return {'message': 'No image URL provided'}
    
    # Run the model
    output = classify_image(img_url)
    
    # Return the results as a dictionary
    return {'predictions': output}