from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load Acne Detection Model (Binary Classification)
acne_model = tf.keras.models.load_model("acne_classification_balanced_model.h5")


def predict_acne(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = acne_model.predict(img_array)[0][0]
    confidence = float(prediction) if prediction >= 0.5 else 1 - float(prediction)
    return "Acne Skin" if prediction < 0.5 else "Clear Skin", confidence


# Load Acne Classification Model (Multi-class)
class_labels = ["Acne Scars", "Blackhead", "Nodules", "Papules", "Pustules", "Whitehead"]


class MultiLabelEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelEfficientNet, self).__init__()
        self.model = models.efficientnet_b3(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


num_classes = len(class_labels)
class_model = MultiLabelEfficientNet(num_classes)
class_model.load_state_dict(torch.load("multi_label_efficientnet_b3.pth", map_location=torch.device("cpu")))
class_model.eval()


def predict_acne_type(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = class_model(img)

    confidence_scores = output.numpy().flatten()
    predicted_labels = [
        {"label": class_labels[i], "confidence": float(confidence_scores[i])}
        for i in range(len(confidence_scores)) if confidence_scores[i] > 0.5
    ]

    return predicted_labels if predicted_labels else [{"label": "No Specific Type Detected", "confidence": 0.0}]


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    img = Image.open(file).convert("RGB")

    acne_result, acne_confidence = predict_acne(img)

    if acne_result == "Acne Skin":
        acne_types = predict_acne_type(img)
    else:
        acne_types = [{"label": "Clear Skin", "confidence": acne_confidence}]

    return jsonify({"result": acne_result, "confidence": acne_confidence, "classification": acne_types})


if __name__ == '__main__':
    app.run(debug=True)
