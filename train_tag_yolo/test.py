import ultralytics, cv2
from pprint import pprint
model = ultralytics.YOLO('/Users/mouyuanyap/Documents/vscode/train_tag_yolo/runs/detect/train4/weights/best.pt')

image = "/Users/mouyuanyap/Documents/vscode/footfallCam_Task/runs/detect/test2/crops/person/labels868.jpg"
results = model.predict(source = image, save=True, save_txt=True, name = "test", exist_ok=True, save_crop = True)

print(results)