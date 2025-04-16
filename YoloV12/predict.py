from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
if __name__ == "__main__":
    
    model = YOLO("path/to/best.pt")

    # Predict with the model
    results = model("test_blended", save=True)  # predict on an image
