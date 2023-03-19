import onnxruntime
import numpy as np
from pytorch_model import Classifier
from PIL import Image

# Define the path to the ONNX model file
MODEL_FILE = "model/model_optimized.onnx"

# Define the input and output node names for the model
INPUT_NODE = "input"
OUTPUT_NODE = "output"



# Define the function to test the ONNX model
def test_onnx_model():

    all_tests_passed = True
    # Create an instance of the ONNX Runtime inference session
    session = onnxruntime.InferenceSession(MODEL_FILE)
    classifier = Classifier() 

    # Get the input and output shapes of the model
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    image_list = ["images/n01440764_tench.jpeg", "images/n01667114_mud_turtle.JPEG"]

    # Test the model on the two example images
    for image_file in image_list:

        img = Image.open(image_file)

        # Preprocess the input image
        img_preprocessed = classifier.preprocess_numpy(img)
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

        output_names = [OUTPUT_NODE]
        input_feed = {INPUT_NODE: img_preprocessed}
        output = session.run(output_names, input_feed)

        # Get the predicted class index
        predicted_classes = np.argmax(output[0], axis=1)

        # Print the predicted class indices
        if image_file == "images/n01440764_tench.jpeg":
            if predicted_classes[0] != 0:
                all_tests_passed = False
                print(f"Test failed: {image_file} belongs to class id {predicted_classes[0]} instead of 0")
            else:
                print(f"Test Passed: {image_file} belongs to class id {predicted_classes[0]}")
        if image_file == "images/n01667114_mud_turtle.JPEG":
            if predicted_classes[0] != 35:
                all_tests_passed = False
                print(f"Test failed: {image_file} belongs to class id {predicted_classes[0]} instead of 35")
            else:
                print(f"Test Passed: {image_file} belongs to class id {predicted_classes[0]}")

        # Raise an exception if any of the tests fail
        if not all_tests_passed:
            raise AssertionError("One or more tests failed")

if __name__ == "__main__":
    test_onnx_model()
