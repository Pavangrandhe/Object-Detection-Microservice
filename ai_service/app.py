from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import json

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # lightweight YOLO model

# Folder to save output images and JSON files
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files['image']
    
    # Read image from uploaded file
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Run detection
    results = model.predict(img)
    
    # Prepare JSON detections
    detections = []
    for box in results[0].boxes.xyxy.tolist():
        detections.append({
            "xmin": box[0],
            "ymin": box[1],
            "xmax": box[2],
            "ymax": box[3]
        })
    
    # Draw bounding boxes on the image
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Unique filename using timestamp
    timestamp = int(time.time() * 1000)
    image_filename = os.path.join(OUTPUT_DIR, f"output_{timestamp}.jpg")
    json_filename = os.path.join(OUTPUT_DIR, f"output_{timestamp}.json")
    
    # Save output image
    cv2.imwrite(image_filename, img)
    
    # Save JSON file
    with open(json_filename, "w") as f:
        json.dump(detections, f, indent=4)
    
    # Include filenames in the response
    response = {
        "detections": detections,
        "output_image": image_filename,
        "json_file": json_filename
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
