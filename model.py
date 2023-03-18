from typing import List
import torch
import onnxruntime
from PIL import Image
from torchvision import transforms
from torch import Tensor
from pytorch_model import Classifier

class Preprocessor:
    def __init__(self, img_size: int = 224) -> None:
        self.img_size = img_size
        self.classifier = Classifier()
        
    def __call__(self, img: Image.Image) -> Tensor:
        img = self.classifier.preprocess_numpy(img)
        return torch.from_numpy(img)

class OnnxModel:
    def __init__(self, onnx_file_path: str) -> None:
        self.session = onnxruntime.InferenceSession(onnx_file_path)

    def predict(self, input_tensor: Tensor) -> List[float]:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        input_feed = {input_name: input_tensor.detach().cpu().numpy()}
        output = self.session.run([output_name], input_feed)
        return output[0].tolist()
