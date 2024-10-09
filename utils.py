import numpy as np


def yolo_label_to_coco(values, w, h):
    # YOLO formatını COCO formatına çevirme
    x_center, y_center, width, height = [np.abs(float(val)) for val in values]
    xmin = int((x_center - width / 2) * w)
    xmax = int((x_center + width / 2) * w)
    ymin = int((y_center - height / 2) * h)
    ymax = int((y_center + height / 2) * h)
    return xmin, ymin, xmax, ymax

def coco_label_to_yolo(xmin, ymin, xmax, ymax, w, h):
    # COCO formatındaki değerleri YOLO formatına çevirme
    width = (xmax - xmin) / w  
    height = (ymax - ymin) / h  
    x_center = (xmin + xmax) / (2 * w) 
    y_center = (ymin + ymax) / (2 * h)  
    
    return x_center, y_center, width, height
