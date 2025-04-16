from ultralytics import YOLO




if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # pass any model type
    
    results = model.train(data="yolo_dataset_blended_N10GM\dataset.yaml", 
                          model="yolo11n.pt",
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
    
    # Perform object detection on an image using the model
    results = model("yolo_dataset_blende_N10GM/test/images")