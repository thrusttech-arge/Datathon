import json
import cv2
import os
import shutil
from sklearn.model_selection import train_test_split
import glob

import utils as ut

with open("annotations/train.json") as f:
    data = json.load(f)
f.close()
annotations = data["annotations"]

if not(os.path.exists("data")):
    os.mkdir("data")

if not(os.path.exists("data/train")):
    os.mkdir("data/train")
    os.mkdir("data/train/images")
    os.mkdir("data/train/labels")

if not(os.path.exists("data/valid")):
    os.mkdir("data/valid")
    os.mkdir("data/valid/images")
    os.mkdir("data/valid/labels")

if not(os.path.exists("data/fullData")):
    os.mkdir("data/fullData")
    os.mkdir("data/fullData/images")
    os.mkdir("data/fullData/labels")

for dataEx in data["images"]:
    imageName = dataEx["file_name"]
    img = cv2.imread(f"images/{imageName}")
    h,w = img.shape[0],img.shape[1]
    imgId = dataEx["id"]

    bboxs = [item for item in annotations if item["image_id"] == imgId]

    bboxsText = ""
    for bbox in bboxs:
        x_min,y_min,width,height = bbox["bbox"]
        x_center, y_center, bbox_width, bbox_height = ut.coco_label_to_yolo(x_min,y_min,int(x_min + width),int(y_min + height),w,h)
        labelCategori = bbox["category_id"]-1
        bboxsText += f"{labelCategori} {x_center} {y_center} {bbox_width} {bbox_height}\n"
    
    labelPath = f"data/fullData/labels/{imageName[:-4]}.txt"
    with open(labelPath,"w") as f:
        f.write(bboxsText)
    f.close()

    imageToPath = f"data/fullData/images/{imageName}"
    shutil.copy(f"images/{imageName}",imageToPath)

trainData,validData = train_test_split(glob.glob("data/fullData/images/**"),test_size=0.33,random_state=42)

for path in trainData:
    
    name = path.split("/")[-1][:-4]
    labelPath = f"data/fullData/labels/{name}.txt"
    labelToPath = f"data/train/labels/{name}.txt"
    imageToPath = f"data/train/images/{name}.png"
    shutil.copy(path,imageToPath)
    shutil.copy(labelPath,labelToPath)

for path in validData:
    name = path.split("/")[-1][:-4]
    labelPath = f"data/fullData/labels/{name}.txt"
    labelToPath = f"data/valid/labels/{name}.txt"
    imageToPath = f"data/valid/images/{name}.png"

    shutil.copy(path,imageToPath)
    shutil.copy(labelPath,labelToPath)

        