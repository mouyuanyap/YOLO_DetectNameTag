import cv2,time
import ultralytics, cv2
from pprint import pprint
from PIL import Image, ImageTk

model = ultralytics.YOLO('yolov8n.pt')
model_NameTag = ultralytics.YOLO('C:\\Users\\User\\Desktop\\pythonCode\\YOLO_DetectNameTag\\train_tag_yolo\\runs\\detect\\train4\\weights\\best.pt')

person_detect_conf = 0.5
nameTag_detect_conf = 0.2

def processResult(image,model, model_NameTag,resultsPerson, flip = False):
    if flip:
        frame1 = image 
        image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        image = cv2.rotate(image, cv2.ROTATE_180)
        image = cv2.rotate(image, cv2.ROTATE_180)
        frame1 = image
    
    height = image.shape[0]
    width = image.shape[1]
    
    for i in range(resultsPerson[0].boxes.cls.size(dim=0)):
        if float(resultsPerson[0].boxes.conf[i].item())> person_detect_conf and model.names[int(resultsPerson[0].boxes.cls[i])] == "person":
            x1 = int(resultsPerson[0].boxes.xyxy[i][0].item())
            y1 = int(resultsPerson[0].boxes.xyxy[i][1].item())
            x2 = int(resultsPerson[0].boxes.xyxy[i][2].item())
            y2 = int(resultsPerson[0].boxes.xyxy[i][3].item())
            
            crop_img = image[y1:y2 , x1:x2]
            result_nameTag = model_NameTag.predict(source = crop_img,save = True, name = "test_tag_{}".format("person"), exist_ok=True)
            if flip:
                x1_new = width - x2
                x2_new = width - x1
                y1_new = height - y2
                y2_new = height - y1
                x1,y1,x2,y2 = x1_new,y1_new,x2_new,y2_new
                frame1 = cv2.rotate(image, cv2.ROTATE_180)
            
            start_point = (x1, y1)
            end_point = (x2,y2)
            # to determine whether has nameTag
            for i in range(result_nameTag[0].boxes.cls.size(dim=0)):
                if float(result_nameTag[0].boxes.conf[i].item())> nameTag_detect_conf and model_NameTag.names[int(result_nameTag[0].boxes.cls[i])] == "tag":
                    frame1 = cv2.rectangle(frame1,start_point,end_point,color=(0,255,0), thickness = 3)
                    if y1 < 30:
                        cv2.putText(frame1, 'x:{}, y:{}'.format(x1,y1), (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    else:
                        cv2.putText(frame1, 'x:{}, y:{}'.format(x1,y1), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    
    return frame1


def video_frame_generator(video_name = "./sample.mp4"):
    def current_time():
        return time.time()

    start_time = current_time()
    _time = 0
    vidObj = cv2.VideoCapture(video_name) 
    frame, image = vidObj.read()
    count = 0
    while frame:
        
        # turn video array into an image and reduce the size
        class_name = 'person'
        
        ori_image = image
        resultsPerson = model.predict(source= image, save=True, name = "test_{}".format(class_name), exist_ok=True)
        image = processResult(image, model, model_NameTag,resultsPerson)
        rotated_image = cv2.rotate(ori_image, cv2.ROTATE_180)
        resultsPerson = model.predict(source= rotated_image, save=True, name = "test_{}_rotated".format(class_name), exist_ok=True)
        image = processResult(image, model, model_NameTag,resultsPerson, True)
        # image = Image.open("./runs/detect/test_person/image0.jpg")
        
        cv2.putText(image, '{}'.format(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)
        image = Image.fromarray(color_coverted) 
        image.thumbnail((750, 750), Image.LANCZOS)

        # make image in a tk Image and put in the label
        outputimage = ImageTk.PhotoImage(image)

        # introduce a wait loop so movie is real time -- asuming frame rate is 24 fps
        # if there is no wait check if time needs to be reset in the event the video was paused
        _time += 1 / 24
        run_time = current_time() - start_time
        while run_time < _time:
            run_time = current_time() - start_time
        else:
            if run_time - _time > 0.1:
                start_time = current_time()
                _time = 0
        frame, image = vidObj.read()
        count +=1
        yield count-1, outputimage


