import ultralytics, cv2
from pprint import pprint
model = ultralytics.YOLO('./train_tag_yolo/runs/detect/train4/weights/best.pt')

image = "./runs/detect/test2/crops/person/labels868.jpg"
results = model.predict(source = image, save=True, save_txt=True, name = "tag_test", exist_ok=True, save_crop = True)

print(results)