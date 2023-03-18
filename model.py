from typing import List
import torch
import onnxruntime
from PIL import Image
from torchvision import transforms
from torch import Tensor


class Preprocessor:
    def __init__(self, img_size: int = 224) -> None:
        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, img: Image.Image) -> Tensor:
        img_tensor = self.transform(img)
        return img_tensor.unsqueeze(0)


class OnnxModel:
    def __init__(self, onnx_file_path: str) -> None:
        self.session = onnxruntime.InferenceSession(onnx_file_path)

    def predict(self, input_tensor: Tensor) -> List[float]:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        input_feed = {input_name: input_tensor.detach().cpu().numpy()}
        output = self.session.run([output_name], input_feed)
        return output[0].tolist()
