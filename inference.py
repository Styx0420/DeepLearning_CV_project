import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import json
import io
from PIL import Image


def model_fn(model_dir):

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 133)
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def input_fn(request_body, content_type):
    if content_type == 'application/x-image':
        img = Image.open(io.BytesIO(request_body)).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0)
        return tensor
    elif content_type == 'application/json':
        image_array = json.loads(request_body)
        tensor = torch.tensor(image_array)
        return tensor
    else:
        raise Exception(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        predicted_class = torch.argmax(outputs, dim=1)
    return predicted_class.item()

def output_fn(prediction, content_type):
    return json.dumps({'predicted_class': prediction})