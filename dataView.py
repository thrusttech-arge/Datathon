import json
import numpy as np
import cv2
import random

with open("annotations/train.json") as f:
    data = json.load(f)

print("hekkı")

category_ids = {item['category_id'] for item in data["annotations"]}
print(f"Farklı category_id sayısı: {len(category_ids)}")
# category_colors = {category_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for category_id in category_ids}

category_colors = [
    (0,0,0),
    (0,0,255),
    (0,255,0),
    (255,0,0)
]

idx = 0
idxEx = -1
while True:
    if idxEx != idx:
        idxEx = idx
        imagePath = data["images"][idx]["file_name"]
        imgId = data["images"][idx]["id"]
        annotations = data["annotations"]
        img = cv2.imread(f"images/{imagePath}")

        bboxs = [item for item in annotations if item["image_id"] == imgId]
        
        for bbox in bboxs:
            x_min,y_min,width,height = bbox["bbox"]
            labelCategori = bbox["category_id"]-1

            img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_min + width), int(y_min + height)), category_colors[labelCategori], 2)

        text = f"Goruntu Id: {idx}"
        img = cv2.putText(img,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
        cv2.imshow("image",img)

    if cv2.waitKey(1) == ord("q"):
        break
    elif cv2.waitKey(1) == ord("d"):
        idx += 1
        if idx > 499:
            idx = 0
    elif cv2.waitKey(1) == ord("a"):
        idx -= 1
        if idx <0:
            idx = 499

cv2.destroyAllWindows()