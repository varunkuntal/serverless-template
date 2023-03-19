import onnxruntime
from PIL import Image
from model import Preprocessor, OnnxModel
from test_server import classify_image
import numpy as np

preprocessor = None
onnx_model = None

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global preprocessor, model
    preprocessor = Preprocessor()
    model = OnnxModel('model/model_optimized.onnx')


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