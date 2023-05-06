import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
import os
import glob



def read_annotations_yolo(annotations_path, image_width, image_height):
    annotations = []
    with open(annotations_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            annotations.append({
                'class_id': class_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
    return annotations

def compute_iou(annot1, annot2):
    """
    Compute the IoU between two bounding boxes.
    
    Arguments:
    annot1 -- a dict of {class_id: int, x1: float, y1: float, x2: float, y2: float} representing the coordinates of the first bounding box
    annot2 -- a dict of {class_id: int, x1: float, y1: float, x2: float, y2: float} representing the coordinates of the second bounding box
    
    Returns:
    iou -- the IoU between the two bounding boxes
    """
    # Calculate the intersection area
    x_left = max(annot1['x1'], annot2['x1'])
    y_top = max(annot1['y1'], annot2['y1'])
    x_right = min(annot1['x2'], annot2['x2'])
    y_bottom = min(annot1['y2'], annot2['y2'])

    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the union area
    annot1_area = (annot1['x2'] - annot1['x1']) * (annot1['y2'] - annot1['y1'])
    annot2_area = (annot2['x2'] - annot2['x1']) * (annot2['y2'] - annot2['y1'])
    union_area = annot1_area + annot2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou

def compute_confusion_matrix(ground_truth_annots, predicted_detections, iou_threshold):
    # Compute the number of true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0
    for image, gt_annotations in ground_truth_annots.items():
        for gt_annot in gt_annotations:
            found_match = False
            if image in predicted_detections.keys():
                for pred_annot in predicted_detections[image]:
                    iou = compute_iou(gt_annot, pred_annot)
                    if iou > iou_threshold:
                        tp += 1
                        found_match = True
                        break
            if not found_match:
                fn += 1
    
    for image, detections in predicted_detections.items():
        for pred_annot in detections:
            found_match = False
            if image in ground_truth_annots.keys():
                for gt_annot in ground_truth_annots[image]:
                    iou = compute_iou(gt_annot, pred_annot)
                    if iou > iou_threshold:
                        found_match = True
                        break
            if not found_match:
                fp += 1

    # Compute the confusion matrix
    confusion_matrix = np.array([[tp, fp], [fn, 0]])
    return confusion_matrix

def compute_precision(tp, fp):
    return tp / (tp + fp)

def compute_recall(tp, fn):
    return tp / (tp + fn)

def display_empiric_confusion_matrix(confusion_matrix, name=''):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['Positive', 'Negative'])
    plt.ylabel('Predicted Label')
    plt.xlabel('True Label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, confusion_matrix[i][j], horizontalalignment="center", color="white" if confusion_matrix[i][j] > 210 else "black")
    plt.show()


def display_probabilistic_confusion_matrix(confusion_matrix, name=''):
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    total_annotations = tp + fn
    probabilistic_confusion_matrix = [[round(tp/total_annotations,3), fp/fp],[round(fn/total_annotations,3), 0]]

    plt.imshow(probabilistic_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Probabilistic onfusion Matrix for {name}')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['Positive', 'Negative'])
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, probabilistic_confusion_matrix[i][j], horizontalalignment="center", color="white" if probabilistic_confusion_matrix[i][j] > 0.5 else "black")
    plt.show()

def compute_bbox_errors(ground_truth_bbox, predicted_bbox):
    # Compute center point of ground truth and predicted bounding boxes
    gt_center_x, gt_center_y = (ground_truth_bbox['x1'] + ground_truth_bbox['x2'])/2, (ground_truth_bbox['y1'] + ground_truth_bbox['y2'])/2
    pred_center_x, pred_center_y = (predicted_bbox['x1'] + predicted_bbox['x2'])/2, (predicted_bbox['y1'] + predicted_bbox['y2'])/2
    
    # Compute width and height of ground truth and predicted bounding boxes
    gt_width, gt_height = ground_truth_bbox['x2'] - ground_truth_bbox['x1'], ground_truth_bbox['y2'] - ground_truth_bbox['y1']
    pred_width, pred_height = predicted_bbox['x2'] - predicted_bbox['x1'], predicted_bbox['y2'] - predicted_bbox['y1']
    
    # Compute error vectors between ground truth and predicted bounding boxes
    cx_error = gt_center_x - pred_center_x
    cy_error = gt_center_y - pred_center_y
    width_error = gt_width - pred_width
    height_error = gt_height - pred_height
    
    return [cx_error, cy_error, width_error, height_error]

def compute_all_bbox_errors(ground_truth_annots, predicted_detections, iou_treshold):
    bbox_error_vectors = []
    for image, annots in ground_truth_annots.items():
        if image in predicted_detections.keys():
            for gt_annot in annots:
                for detection in predicted_detections[image]:
                    iou = compute_iou(gt_annot, detection)
                    if iou > iou_treshold:
                        error_vector = compute_bbox_errors(gt_annot, detection)
                        bbox_error_vectors.append(error_vector)
    return bbox_error_vectors

def calculate_varinace(numbers):
    # Step 1: Calculate the expected value
    expected_value = sum(numbers) / len(numbers)
    #expected_value = 0


    # Step 2: Subtract the expected value
    differences = [elem - expected_value for elem in numbers]

    # Step 3: Square the differences
    squared_differences = [diff**2 for diff in differences]

    # Step 4: Calculate the sum of squared differences
    sum_squared_diff = sum(squared_differences)

    # Step 5: Divide by the number of elements to get the variance
    variance = sum_squared_diff / len(numbers)


    print("Expected value:", expected_value)
    print("Variance:", variance)

    return expected_value, variance

def dropout_per_image(ground_truth_annots, predicted_detections, iou_threshold):
    # Compute the number of true positives, false positives, and false negatives
    dropout_images = {}
    fn = 0
    for image, gt_annotations in ground_truth_annots.items():
        fn = 0
        for gt_annot in gt_annotations:
            found_match = False
            if image in predicted_detections.keys():
                for pred_annot in predicted_detections[image]:
                    iou = compute_iou(gt_annot, pred_annot)
                    if iou > iou_threshold:
                        found_match = True
                        break
            if not found_match:
                fn += 1
        dropout_images[int(image)] = fn/len(gt_annotations)
        
    return dropout_images

def read_annotations_synthetic(annotations_path, image_width, image_height):

    annotations = []
    with open(annotations_path, 'r') as data_file:
        json_data = data_file.read()
    data = json.loads(json_data)
    for i, annot in enumerate(data):
        class_id = 0
        bbox = annot['BB2D']

        # xmin, ymin, xmax, ymax
        # Cut bbox so that it is inside the image. 
        x_min = max(bbox[0]['X'],0)
        y_min = max(bbox[0]['Y'],0)
        x_max = min(bbox[1]['X'],image_width)
        y_max = min(bbox[1]['Y'],image_height)

        if x_max < 0 or y_max<0:
            continue
        if x_min > image_width or y_min > image_height:
            continue

        annotations.append({
                'class_id': class_id,
                'x1': x_min,
                'y1': y_min,
                'x2': x_max,
                'y2': y_max
            })
    return annotations


def dropout_per_image_synthetic_dataset(ground_truth_annots, predicted_detections, iou_threshold):
    # Compute the number of true positives, false positives, and false negatives
    dropout_images = [{},{},{}]

    fn = 0
    for image, gt_annotations in ground_truth_annots.items():
        camera_number = int(image[3:5])
        fn = 0
        for gt_annot in gt_annotations:
            found_match = False
            if image in predicted_detections.keys():
                for pred_annot in predicted_detections[image]:
                    iou = compute_iou(gt_annot, pred_annot)
                    if iou > iou_threshold:
                        found_match = True
                        break
            if not found_match:
                fn += 1
        if len(gt_annotations)>0:
            dropout_images[camera_number][int(image[-3:])] = fn/len(gt_annotations)
        else:
            dropout_images[camera_number][int(image[-3:])] = 0
        
    return dropout_images



def get_file_name_synthetic(path):
    return path.strip().split('/')[-1].split('.')[0]

def display_stats(ground_truth_annots, predicted_detections, iou_treshold, name=''):
    confusion_matrix = compute_confusion_matrix(ground_truth_annots, predicted_detections, iou_treshold)
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]

    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)
    print('Displaying stats')
    # Display the results
    print("Confusion matrix:")
    print(confusion_matrix)
    print("Precision:")
    print(precision)
    print("Recall:")
    print(recall)
    display_empiric_confusion_matrix(confusion_matrix, name=name)
    display_probabilistic_confusion_matrix(confusion_matrix, name=name)
    print(f'\n\nBOUNDING BOX ERRORS FOR {name}')

    bbox_error_vectors = compute_all_bbox_errors(ground_truth_annots, predicted_detections, iou_treshold)

    cx_e = [x[0] for x in bbox_error_vectors]
    cy_e = [x[1] for x in bbox_error_vectors]
    width_e = [x[2] for x in bbox_error_vectors]
    height_e = [x[3] for x in bbox_error_vectors]

    print('IoU threshold: ', iou_treshold)
    print('Error of center x')
    cx_expected_value, cx_variance = calculate_varinace(cx_e)
    print('\nError of center y')
    cy_expected_value, cy_variance = calculate_varinace(cy_e)
    print('\nError of width')
    width_expected_value, width_variance = calculate_varinace(width_e)
    print('\nError of height')
    height_expected_value, height_variance = calculate_varinace(height_e)


    print('\n\n DROPOUT STATS')
    dropout_images_all_cams = dropout_per_image_synthetic_dataset(ground_truth_annots, predicted_detections, iou_treshold)
    for i, dropout_images in enumerate(dropout_images_all_cams):
        sorted_image_numbers = list(dropout_images.keys())
        sorted_image_numbers.sort()
        sorted_dropout = [dropout_images[image_number] for image_number in sorted_image_numbers]
        plt.title(f'Dropout per image for {name} with camera {i}')
        plt.plot(sorted_image_numbers, sorted_dropout)
        plt.show()

def get_annots_and_detections(GROUND_TRUTH_PATHS, PREDICTED_PATH, IMAGE_WIDTH, IMAGE_HEIGHT):
    image_paths = list(itertools.chain.from_iterable([glob.glob(f'{path}/images/*.json') for path in GROUND_TRUTH_PATHS]))
    ground_truth_annots = {get_file_name_synthetic(path): read_annotations_synthetic(path, IMAGE_WIDTH, IMAGE_HEIGHT) for path in image_paths}

    predicted_annot_paths = [os.path.join(PREDICTED_PATH, filename) for filename in os.listdir(PREDICTED_PATH)]
    predicted_detections = {get_file_name_synthetic(image): read_annotations_yolo(image, IMAGE_WIDTH, IMAGE_HEIGHT) for image in predicted_annot_paths}
    return ground_truth_annots, predicted_detections

def display_stats_main(GROUND_TRUTH_PATHS, PREDICTED_PATH, iou_treshold, IMAGE_WIDTH, IMAGE_HEIGHT, name=''):
    ground_truth_annots, predicted_detections = get_annots_and_detections(GROUND_TRUTH_PATHS, PREDICTED_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)
    display_stats(ground_truth_annots, predicted_detections, iou_treshold, name=name)

