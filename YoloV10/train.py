from ultralytics import YOLO


if __name__ == "__main__":
    # Load a pre-trained YOLOv10n model
    model = YOLO("yolov10n.pt")

    # Display model information (optional)
    model.info()

    # Train the model
    results = model.train(data="yolo_dataset_blended_N10GM\dataset.yaml", 
                          model="yolov10n.pt",
                          epochs=150, 
                          batch=8, 
                          imgsz=1024,
                          scale=0.5,  
                          mosaic=1.0,
                          mixup=0.0,  
                          copy_paste=0.1,
                          device="0",
                          patience=50)
                        
    # Evaluate the model's performance on the validation set
    results = model.val()
    
    # Run inference with the YOLOv9c model on the 'bus.jpg' image
    results = model("yolo_dataset_blended_N10GM/test/images")