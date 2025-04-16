from ultralytics import YOLO


if __name__ == "__main__":
    
    model = YOLO("path/to/best.pt")
    
    # Predict with the model
    results = model("test_blended", save=True)  # predict on an image

 