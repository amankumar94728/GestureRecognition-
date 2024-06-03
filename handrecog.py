import cv2
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Load pre-trained ViT model and feature extractor
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Open the video file
video_path = "./input/aman.mp4"
cap = cv2.VideoCapture(video_path)

# Define the classes for hand gestures
classes = ["Fist", "Open Hand", "Thumb Up", "Thumb Down"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    inputs = feature_extractor(images=frame, return_tensors="pt")
    
    # Perform inference
    outputs = model(**inputs)
    predicted_class_idx = torch.argmax(outputs.logits).item()
    predicted_class = classes[predicted_class_idx]
    
    # Display the prediction
    cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
