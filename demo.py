import ultralytics, cv2
from pprint import pprint
model = ultralytics.YOLO('yolov8n.pt')
model.to('cuda')
model_NameTag = ultralytics.YOLO('C:\\Users\\User\\Desktop\\pythonCode\\YOLO_DetectNameTag\\train_tag_yolo\\runs\\detect\\train4\\weights\\best.pt')
model_NameTag.to('cuda')


def outputFrame(frame, count, results, class_name = "", flip = ""):
    # to determine person object
    frame1 = frame
    for i in range(results[0].boxes.cls.size(dim=0)):
        
        if float(results[0].boxes.conf[i].item())>0.5 and model.names[int(results[0].boxes.cls[i])] == class_name:
            print(count)
            print("Classes: {}, {} - {}".format(model.names[int(results[0].boxes.cls[i])],results[0].boxes.conf[i],results[0].boxes.xyxy[i]))
            x1 = int(results[0].boxes.xyxy[i][0].item())
            y1 = int(results[0].boxes.xyxy[i][1].item())
            x2 = int(results[0].boxes.xyxy[i][2].item())
            y2 = int(results[0].boxes.xyxy[i][3].item())
            start_point = (x1, y1)
            end_point = (x2,y2)
            crop_img = frame[y1:y2 , x1:x2]
            result_nameTag = model_NameTag.predict(source = crop_img,save = True, name = "test_tag_{}{}".format(class_name,flip), exist_ok=True)
            
            # to determine whether has nameTag
            
            for i in range(result_nameTag[0].boxes.cls.size(dim=0)):
                if float(result_nameTag[0].boxes.conf[i].item())>0.3 and model_NameTag.names[int(result_nameTag[0].boxes.cls[i])] == "tag":
                    frame1 = cv2.rectangle(frame1,start_point,end_point,color=(0,255,0), thickness = 2)

    cv2.imwrite('C:\\Users\\User\\Desktop\\pythonCode\\YOLO_DetectNameTag\\frame1.png',frame1)
        

vidObj = cv2.VideoCapture("sample.mp4") 
success, frame = vidObj.read()
# path = 'test.png'
count = 0
while success:
    if count >850:
        
        class_name = 'person'
        results = model.predict(source= frame, save=True, name = "test_{}".format(class_name), exist_ok=True)
        outputFrame(frame,count,results,class_name)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # results = model.predict(source= frame, save=True, save_txt=True, name = "test_{}_flip".format(class_name), exist_ok=True, save_crop = True)
        # outputFrame(frame,count,class_name,"_flip")
    success, frame = vidObj.read()
    count +=1
# results = model(image, classes = 0)



# image = cv2.imread(path)
# image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
# image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# cv2.imshow(path,image)
# cv2.waitKey(0)