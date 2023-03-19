import requests
import json
import argparse
import time
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
model_key = os.getenv("MODEL_KEY")

# Set up the command-line arguments
parser = argparse.ArgumentParser(description="Test deployment on the Banana dev server.")
parser.add_argument("image_path", help="Path or URL of the image to classify.")
parser.add_argument("-t", "--test", help="Run preset custom tests.", action="store_true")
args = parser.parse_args()

# Set up the API request data
image_path = args.image_path
data = {
    "apiKey": api_key,
    "modelKey": model_key,
    "modelInputs": {
        "url": image_path
    }
}
json_data = json.dumps(data)

# Send the API request and time the call
start_time = time.time()
url = "https://api.banana.dev/start/v4/"
headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers, data=json_data)
end_time = time.time()
call_time = end_time - start_time

# Parse the prediction result
result = json.loads(response.content.decode('utf-8'))
prediction = result['modelOutputs']

# Print the prediction and call time
print(f"Prediction: {prediction}")
print(f"Call time: {call_time} seconds")

# Run preset custom tests, if specified
if args.test:
    # Set up the test images and expected results
    test_images = ["images/n01440764_tench.jpeg",
                   "images/n01667114_mud_turtle.JPEG"]
    expected_results = ["tench", "mud turtle"]

    # Loop through the test images and compare the results to the expected results
    for i, image_url in enumerate(test_images):
        # Set up the API request data
        data = {
            "apiKey": api_key,
            "modelKey": model_key,
            "modelInputs": {
                "url": image_url
            }
        }
        json_data = json.dumps(data)

        # Send the API request and time the call
        start_time = time.time()
        response = requests.post(url, headers=headers, data=json_data)
        end_time = time.time()
        call_time = end_time - start_time

        # Parse the prediction result
        result = json.loads(response.content.decode('utf-8'))
        prediction = result['modelOutputs'][0]['predictions']


        # Compare the prediction to the expected result
        if prediction == expected_results[i]:
            print(f"Test {i+1}: PASSED")
        else:
            print(f"Test {i+1}: FAILED")
        print(f"Prediction: {prediction}")
        print(f"Call time: {call_time} seconds")
