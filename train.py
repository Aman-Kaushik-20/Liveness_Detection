from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

model.train(data='Dataset/SplitData/data.yaml', epochs=3)  # train the model