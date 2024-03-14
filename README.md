# YOLOv8 Object Detection - Staff Name Tag

## Table of Contents
1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Object Detection Pipeline](#4-object-detection-pipeline)
5. [Version History](#5-version-history)
6. [References](#6-references)

## 1. Introduction
This project implements real-time object detection using a two-stage approach with pre-trained and fine-tuned YOLOv8 models. Initially, it detects person objects within frames of videos, followed by determining whether the person is wearing a custom name tag. This streamlined process enables efficient and accurate identification of individuals and their associated attributes in real-time scenarios.

## 2. Installation
1. Clone this repository `git clone https://github.com/mouyuanyap/YOLO_DetectNameTag.git`.
2. Create a new virtual environment and activate it.
3. Install required libraries using `pip install -r requirements.txt`.

> [!IMPORTANT]
> Required Python 3.11 or higher to run Tkinter GUI on MacOS.

## 3. Usage
### CLI command to detect video frames and output realtime result on GUI
```
python demo_gui.py "./sample.mp4"
```
Frame with name tag detected is saved in result/frameXX_(x,y)_conf.png
###

## 4. Object Detection Pipeline
### Use Pretrained YOLOv8 model to detect person object in video frames.
Detect and crop out person objects for further annotation and finetuning.
https://github.com/mouyuanyap/YOLO_DetectNameTag/blob/c5fbe6605f3cfbf84afe72a865800be84fdfe5bb/processFrame.py#L72-L74

### Annotate name tag label using labelImg
![Screenshot of annotating using labelImg](https://github.com/mouyuanyap/YOLO_DetectNameTag/blob/9cb9ab9f044e9db2ace3408183ee8cfb1e441fa3/docs/Screenshot-labelImg.png)

### Finetune YOLOv8 model with custom dataset
Train dataset: 41, Validation dataset: 13, with data augmentation; 50 epochs 
https://github.com/mouyuanyap/YOLO_DetectNameTag/blob/12f1679f8f0f2802a9f4d7bd60770964989e1863/train_tag_yolo/train.py#L1-L5
![Finetune training result](https://github.com/mouyuanyap/YOLO_DetectNameTag/blob/fd9148736d00c645373aec2ba8d944f82e770d32/train_tag_yolo/runs/detect/train4/results.png)

### Two-stage Detection 
1. Detect all person objects in the frame
2. Detect whether name tag object is in each cropped person objects

<img align="center" width="75%" src="https://github.com/mouyuanyap/YOLO_DetectNameTag/blob/12f1679f8f0f2802a9f4d7bd60770964989e1863/docs/GUI-detection-video.gif">

> [!TIP]
> Pretrained YOLOv8 model only can detect person in upright position.
> Hence, rotate the frame clockwise 180 degree and do the above two-stage detection for each frame.

<img src="https://github.com/mouyuanyap/YOLO_DetectNameTag/blob/12f1679f8f0f2802a9f4d7bd60770964989e1863/docs/Screenshot-YOLO-inference-norotate.jpg" width="48%" > <img src="https://github.com/mouyuanyap/YOLO_DetectNameTag/blob/12f1679f8f0f2802a9f4d7bd60770964989e1863/docs/Screenshot-YOLO-inference-rotate.jpg" width="48%" >



## 5. Version History
- **v1.0.0 (2024-03-14)**: Initial release.
- 
## 6. References
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [HumanSignal labelImg](https://github.com/HumanSignal/labelImg)
