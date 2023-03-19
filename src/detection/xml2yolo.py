import xml.etree.ElementTree as ET
import glob
import os
import json
import cv2

# Inspired by: https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
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


def xml_to_yolo(input_dir, output_dir, image_dir, classes):
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
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpeg")):
            print(f"{filename} image does not exist!")
            continue

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
                continue
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


def formate_output(input_dir, output_file, image_dir):
    """For each image in the test dataset, your algorithm needs to predict a list of labels, and the corresponding bounding boxes. The output is expected to contain the following two columns:
    ImageId: the id of the test image, for example, India_00001
    PredictionString: the prediction string should be space-delimited of 5 integers. 
    For example, 2 240 170 260 240 means it's label 2, with a bounding box of coordinates 
    (x_min, y_min, x_max, y_max). We accept up to 5 predictions. For example, 
    if you submit 3 42 24 170 186 1 292 28 430 198 4 168 24 292 190 5 299 238 443 374 2 160 195 294 357 6 1 224 135 356 which contains 6 bounding boxes, we will only take the first 5 into consideration."""

    files = glob.glob(os.path.join(input_dir, '*.txt'))

    number_to_correct_number_dict = {'0': 2, '1': 4, '2':1, '3':3}
    #if not os.path.exists(output_file):
    #    os.mkdir(output_file)
    for file in files:
        basename = os.path.basename(file)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
            print(f"{filename} image does not exist!")
            continue
        image = cv2.imread(os.path.join(image_dir, f"{filename}.jpg"))
        height, width = image.shape[:2]
        result = []
        confidence = []
        with open(file, 'r') as f1:
            lines = f1.readlines()
            for line in lines:
                bb_result = []
                elms = line.strip('\n').split(' ')
                bb_result.append(number_to_correct_number_dict[str(elms[0])])
                bb = yolo_to_xml_bbox([float(x) for x in elms[1:-1]], width, height)
                for el in bb:
                    bb_result.append(el)
                result.append(bb_result)
                confidence.append(elms[-1])
        if len(result) > 5:
            indices = sorted(range(len(confidence)), key=lambda k: confidence[k])
            sorted_results = []
            for i in range(5):
                sorted_results.append(result[indices[i]])
            result = sorted_results
        with open(output_file, 'a+') as f2:
            f2.write(filename + ".jpg,")
            f2.write(" ".join(str(x) for y in result for x in y) + "\n")  
    
    reformat_annot(output_file)


def reformat_annot(path):
    # Find ids
    ids = []
    with open(path, 'r') as f:
        for line in f.readlines():
            print(line)
            ide = line.strip('Norway_').strip('\n')
            id = ide.split(',')[0]
            i = id.split('.')[0]
            ids.append(int(i))
    # Find missing images which are images w0 lables
    add_me = []
    for i in range(8161, 10201):
        if not i in ids:
            add_me.append(i)
    
    # Make new list
    new_list = ['Norway_0'+str(i)+'.jpg,' if i>9999 else 'Norway_00'+str(i)+'.jpg,' for i in add_me]

    # Add images
    with open(path, 'a+') as f:
        f.write('\n')
        for id in new_list:
            f.write(id +'\n')
    
    all_ids = []
    all_line = []
    with open(path, 'r') as f:
        for line in f.readlines():
            all_line.append(line)
            ide = line.strip('Norway_').strip('\n')
            id = ide.split(',')[0]
            i = id.split('.')[0]
            all_ids.append(int(i))

    indices = sorted(range(len(all_ids)), key=lambda k: all_ids[k])
    sorted_lines = [all_line[i] for i in indices]

    with open(path, 'w+') as f:
        for line in sorted_lines:
            f.write(line)
    
    all_lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            print(line)
            elms = line.strip('\n').split(',')
            full_str = elms[0]+','+'\n'
            if elms[1]!='':
                elms2 = elms[1].split(' ')
                print(elms2)
                integer = [str(int(float(x))) for x in elms2]
                int_str = ' '.join(integer)
                full_str = elms[0] + ',' + int_str +'\n'
            all_lines.append(full_str)
    
    with open(path, 'w+') as f:
        for line in all_lines:
            f.write(line)