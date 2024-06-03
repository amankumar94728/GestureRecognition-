import cv2
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Load pre-trained ViT model and feature extractor
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Load the input image
image_path = "./input/amanji.png"  

# Read the image
frame = cv2.imread(image_path)

# Check if the image was loaded successfully
if frame is None:
    print("Error: Unable to load the image. Please check the image path.")
else:
    # Define the classes for hand gestures
    classes = ["Fist", "Open Hand", "Thumb Up", "Thumb Down"]

    # Preprocess the image
    inputs = feature_extractor(images=frame, return_tensors="pt")
        
    # Perform inference
    outputs = model(**inputs)
    predicted_class_idx = torch.argmax(outputs.logits).item()
    predicted_class = classes[predicted_class_idx]
        
    # Print the detected gesture
    print("-------------OUTPUT---------------")
    print("Detected Gesture:", predicted_class)
