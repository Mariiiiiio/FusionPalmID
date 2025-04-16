from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov12n.pt')

    # Train the model
    results = model.train(
    data="yolo_dataset_blended_N10GM/dataset.yaml",
    model="yolov12n.pt",
    epochs=150, 
    batch=8, 
    imgsz=1024,
    scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
    copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
    device="0",
    patience=50,
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("yolo_dataset_blende_N10GM/test/images")
    results[0].show()
