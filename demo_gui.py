
import tkinter as tk
from processFrame import video_frame_generator
import sys
# model = ultralytics.YOLO('yolov8n.pt')
# model.to('cuda')
# model_NameTag = ultralytics.YOLO('C:\\Users\\User\\Desktop\\pythonCode\\YOLO_DetectNameTag\\train_tag_yolo\\runs\\detect\\train4\\weights\\best.pt')
# model_NameTag.to('cuda')

global movie_frame
global pause_video

def _stop():
    global pause_video
    pause_video = True


def _start():
    global pause_video
    pause_video = False

def _replay():
    global pause_video
    pause_video = True
    global movie_frame
    movie_frame = video_frame_generator()
    pause_video = False



if __name__ == "__main__":
    try:
        video_name = sys.argv[1]
    except:
        video_name = "./sample.mp4"
    root = tk.Tk()
    root.title('Name Tag Detector')



    my_label = tk.Label(root)
    my_label.pack()
    tk.Button(root, text='start', command=_start).pack(side=tk.LEFT)
    tk.Button(root, text='stop', command=_stop).pack(side=tk.LEFT)
    tk.Button(root, text='replay', command=_replay).pack(side=tk.LEFT)

    pause_video = False
    movie_frame = video_frame_generator(video_name)
    
    while True:
        if not pause_video:
            try:
                frame_number, frame = next(movie_frame)
                my_label.config(image=frame)
            except:
                _replay()
        root.update()
    root.mainloop()
    