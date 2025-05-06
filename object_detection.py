# Step 1: Import Libraries
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt

# Step 2: Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 3: Load Pretrained YOLOv8 Model
model = YOLO("yolov8n.pt")  # Or yolov8s.pt, etc.

# Step 4: Train on Local Dataset
# Replace with the actual path to your data.yaml
data_yaml_path = "/path/to/your_dataset/data.yaml"

results = model.train(
    data=data_yaml_path,
    epochs=30,
    imgsz=416,
    batch=16,
    device=device
)

# Print training metrics
print("Precision:", results.metrics['metrics/precision(B)'])
print("Recall:", results.metrics['metrics/recall(B)'])
print("mAP50:", results.metrics['metrics/mAP50(B)'])
print("mAP50-95:", results.metrics['metrics/mAP50-95(B)'])

# Step 5: Save the Best Model
model_path = "best_yolov8_model.pt"
model.save(model_path)

# Step 6: Load Trained Model for Inference
model = YOLO(model_path)

# Step 7: Run Inference on an Image
image_path = "test.jpg"  # Replace with your image
results = model(image_path, conf=0.25)

# Step 8: Display the Result
annotated_frame = results[0].plot()
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detection Result")
plt.show()
