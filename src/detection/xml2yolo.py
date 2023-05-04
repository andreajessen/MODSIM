import xml.etree.ElementTree as ET
import glob
import os
import json
import cv2

# Inspired by: https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = round(((bbox[2] + bbox[0]) / 2) / w,5)
    y_center = round(((bbox[3] + bbox[1]) / 2) / h, 5)
    width = round((bbox[2] - bbox[0]) / w, 5)
    height = round((bbox[3] - bbox[1]) / h, 5)
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = float((bbox[0] * w) - w_half_len)
    ymin = float((bbox[1] * h) - h_half_len)
    xmax = float((bbox[0] * w) + w_half_len)
    ymax = float((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def xml_to_yolo(input_dir, output_dir, image_dir, classes, merge_classes=False):
    # create the labels folder (output directory)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(input_dir, '*.xml'))
    # loop through each 
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        """if not os.path.exists(os.path.join(image_dir, f"{filename}.jpeg")):
            if not os.path.exists(os.path.join(image_dir, f"{filename}.png")):
                print(f"{filename} image does not exist!")
                continue"""

        result = []

        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # If class is not a valid class, ignore remove the annotation
            if label not in classes:
                # NB: HACK TO MERGE CLASSES
                if not merge_classes:
                    continue
            if merge_classes:
                index = 0
            else: 
                index = classes.index(label)
            pil_bbox = [float(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

    # generate the classes file as reference
    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))