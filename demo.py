import ultralytics, cv2
from pprint import pprint
model = ultralytics.YOLO('yolov8n.pt')


vidObj = cv2.VideoCapture("sample.mp4") 
success, frame = vidObj.read()

# path = 'test.png'
count = 0
while success:
    if count >500:
        
        results = model.predict(source= frame, save=True, save_txt=True, name = "test2", exist_ok=True, save_crop = True)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        results = model.predict(source= frame, save=True, save_txt=True, name = "test", exist_ok=True, save_crop = True)
    success, frame = vidObj.read()
    count +=1
# results = model(image, classes = 0)



# image = cv2.imread(path)
# image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
# image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# cv2.imshow(path,image)
# cv2.waitKey(0)