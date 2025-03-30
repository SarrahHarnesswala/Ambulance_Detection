import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ========================
# Load Classification Model
# ========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load trained classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("ambulance_classifier.pth", map_location=device))
model.eval()

# ========================
# Load YOLOv8 Model
# ========================
yolo_model = YOLO("yolov8n.pt")

# ========================
# Define Transforms
# ========================
val_test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========================
# Streamlit UI
# ========================
st.title("Ambulance Detection & Classification")
st.write("Upload an image to detect and classify ambulances.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    
    # Run YOLO detection
    results = yolo_model(image_np)
    
    # Define relevant class IDs
    bus_like_classes = [5, 7, 8, 9]  # bus, truck, fire truck, ambulance
    confidence_threshold = 0.30
    detected_vehicles = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            if class_id in bus_like_classes and confidence >= confidence_threshold:
                bbox = box.xyxy[0].tolist()
                detected_vehicles.append((class_id, bbox, confidence))
    
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    
    if detected_vehicles:
        for class_id, bbox, confidence in detected_vehicles:
            x1, y1, x2, y2 = map(int, bbox)
            cropped_img = image_np[y1:y2, x1:x2]
            input_tensor = val_test_transforms(Image.fromarray(cropped_img)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            _, pred = torch.max(output, 1)
            class_label = "Ambulance" if pred.item() == 1 else "Not Ambulance"
            
            st.image(cropped_img, caption=f"Detected: {class_label} (**Confidence: {confidence:.2f}**) ", use_column_width=True)
    else:
        st.write("No bus-like vehicles detected with confidence ≥ 50%.")
