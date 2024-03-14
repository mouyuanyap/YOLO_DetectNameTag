from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

results = model.train(data='./train_tag_yolo/dataset.yaml', epochs=50, imgsz=640,batch=8)

