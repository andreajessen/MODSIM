import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
import os
import glob
import math

from PIL import Image
import matplotlib.patches as patches


def read_annotations_yolo(annotations_path, image_width, image_height):
    annotations = []
    if not os.path.exists(annotations_path): 
        # No annotations in iamge
        print('No annotations in ', annotations_path)
        return annotations
    with open(annotations_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            if len(parts) > 5:
                confidence = float(parts[5])
            else:
                confidence = None

            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            annotations.append({
                'class_id': class_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'confidence': confidence
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
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def compute_recall(tp, fn):
    if tp + fn == 0:
        return 0
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
    return probabilistic_confusion_matrix

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
    bb_err_to_img = {}
    for image, annots in ground_truth_annots.items():
        bb_err_to_img[image] = []
        if image in predicted_detections.keys():
            for gt_annot in annots:
                for detection in predicted_detections[image]:
                    iou = compute_iou(gt_annot, detection)
                    if iou > iou_treshold:
                        error_vector = compute_bbox_errors(gt_annot, detection)
                        bbox_error_vectors.append(error_vector)
                        bb_err_to_img[image].append(error_vector)
    return bbox_error_vectors, bb_err_to_img

def calculate_expected_value_and_std(numbers):
    # Step 1: Calculate the expected value
    expected_value = sum(numbers) / len(numbers)
    #expected_value = 0

    # Plotting the histogram
    plt.hist(numbers, bins=200, edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Error')
    plt.grid(True)

    # Display the histogram
    plt.show()
    # Step 2: Subtract the expected value
    differences = [elem - expected_value for elem in numbers]
    max_diff = differences.index(max(differences))
    print(max_diff)

    # Step 3: Square the differences
    squared_differences = [diff**2 for diff in differences]

    # Step 4: Calculate the sum of squared differences
    sum_squared_diff = sum(squared_differences)

    # Step 5: Divide by the number of elements to get the variance
    variance = sum_squared_diff / (len(numbers)-1)

    # Plot normal distribution
    x = np.linspace(-variance, variance, 100)  # Adjust the range as needed

    # Calculate the probability density function (PDF) for each point
    pdf = 1 / np.sqrt(2 * np.pi * variance) * np.exp(-(x - expected_value)**2 / (2 * variance))
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    # Plotting the normal distribution
    plt.plot(x, pdf, color='blue')

    plt.title(f'Normal Distribution (mean={expected_value}, variance={variance})')

    # Display the plot
    plt.show()

    std = math.sqrt(variance)
    print("Expected value:", expected_value)
    print("Standard deviation:", std)

    return round(expected_value,3), round(std,3)

def dropout_per_image(ground_truth_annots, predicted_detections, iou_threshold):
    # Compute the number of true positives, false positives, and false negatives
    dropout_images = {}
    fn = 0
    for image, gt_annotations in ground_truth_annots.items():
        if len(gt_annotations) == 0:
            dropout_images[int(image)] = 0
        else:
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
    if not os.path.exists(annotations_path): return annotations
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

def filter_detections_by_confidence(detections, confidence_threshold):
    filtered_detections = {}
    for image_name, image_detections in detections.items():
        filtered_image_detections = []
        for detection in image_detections:
            if detection['confidence'] >= confidence_threshold:
                filtered_image_detections.append(detection)
        if len(filtered_image_detections) > 0:
            filtered_detections[image_name] = filtered_image_detections
    return filtered_detections

def create_precision_recall_curve(ground_truth_annots, predicted_detections, iou_threshold):
    precisions = []
    recalls = []
    cf_thresholds = np.arange(0.0, 1.0, 0.05)
    for confidence_threshold in cf_thresholds:
        predicted_detections_above_confidence = filter_detections_by_confidence(predicted_detections, confidence_threshold)
        confusion_matrix = compute_confusion_matrix(ground_truth_annots, predicted_detections_above_confidence, iou_threshold)
        tp = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        print(confidence_threshold, tp, fp, fn)
        precisions.append(compute_precision(tp, fp))
        recalls.append(compute_recall(tp, fn))
    print(precisions)
    print(recalls)
    plt.plot(recalls, precisions, 'b', label='Precision-Recall Curve')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    plt.plot(cf_thresholds, precisions, 'b', label='Precision-Recall Curve')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.xlabel('Confidence')
    plt.ylabel('Precision')
    plt.show()
    plt.plot(cf_thresholds, recalls, 'b', label='Precision-Recall Curve')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.xlabel('Confidence')
    plt.ylabel('Recall')
    plt.show()



def get_file_name_synthetic(path):
    return path.strip().split('/')[-1].split('.')[0]

def display_stats(ground_truth_annots, predicted_detections, iou_treshold, confidence_threshold, name=''):
    #create_precision_recall_curve(ground_truth_annots, predicted_detections, iou_treshold)
    predicted_detections = filter_detections_by_confidence(predicted_detections, confidence_threshold)
    confusion_matrix = compute_confusion_matrix(ground_truth_annots, predicted_detections, iou_treshold)
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    false_discovery_rate = fp/(fp+tp)

    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)
    f1_score = 2*(precision*recall/(precision+recall))
    print('Displaying stats')
    # Display the results
    print("Confusion matrix:")
    print(confusion_matrix)
    print("Precision:")
    print(precision)
    print("Recall:")
    print(recall)
    display_empiric_confusion_matrix(confusion_matrix, name=name)
    probabilistic_confusion_matrix = display_probabilistic_confusion_matrix(confusion_matrix, name=name)
    print(f'{round(precision,3)} & {round(recall,3)} & {round(f1_score,3)} & {round(probabilistic_confusion_matrix[1][0],3)} & {round(false_discovery_rate,3)}')
    print(f'\n\nBOUNDING BOX ERRORS FOR {name}')

    bbox_error_vectors, bb_err_to_img = compute_all_bbox_errors(ground_truth_annots, predicted_detections, iou_treshold)

    cx_e = [x[0] for x in bbox_error_vectors]
    cy_e = [x[1] for x in bbox_error_vectors]
    width_e = [x[2] for x in bbox_error_vectors]
    height_e = [x[3] for x in bbox_error_vectors]

    print('IoU threshold: ', iou_treshold)
    covariance_matrix = np.cov(bbox_error_vectors, rowvar=False)
    print(covariance_matrix)
    print('Error of center x')
    cx_expected_value, cx_std = calculate_expected_value_and_std(cx_e)
    print('\nError of center y')
    cy_expected_value, cy_std = calculate_expected_value_and_std(cy_e)
    print('\nError of width')
    width_expected_value, width_std = calculate_expected_value_and_std(width_e)
    print('\nError of height')
    height_expected_value, height_std = calculate_expected_value_and_std(height_e)
    expected_value_vector = [cx_expected_value, cy_expected_value, width_expected_value, height_expected_value]
    print(expected_value_vector)
    print(f'{cx_expected_value} & {cx_std} & {cy_expected_value} & {cy_std} & {width_expected_value} & {width_std} & {height_expected_value} & {height_std}')
    

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
    #image_paths = list(itertools.chain.from_iterable([glob.glob(f'{path}/images/*.json') for path in GROUND_TRUTH_PATHS]))

    image_paths = []
    for folder in GROUND_TRUTH_PATHS:
        with open(os.path.join(folder, 'test.txt'), "r") as file:
            paths = file.read().splitlines()

        image_paths = [path.strip().strip('.jpg')+'.json' for path in paths]
    
    ground_truth_annots = {get_file_name_synthetic(path): read_annotations_synthetic(path, IMAGE_WIDTH, IMAGE_HEIGHT) for path in image_paths}

    predicted_annot_paths = [os.path.join(PREDICTED_PATH, filename) for filename in os.listdir(PREDICTED_PATH)]
    predicted_detections = {get_file_name_synthetic(image): read_annotations_yolo(image, IMAGE_WIDTH, IMAGE_HEIGHT) for image in predicted_annot_paths}
    return ground_truth_annots, predicted_detections

def display_stats_main(GROUND_TRUTH_PATHS, PREDICTED_PATH, iou_treshold, CONFIDENCE_THRESHOLD, IMAGE_WIDTH, IMAGE_HEIGHT, name=''):
    ground_truth_annots, predicted_detections = get_annots_and_detections(GROUND_TRUTH_PATHS, PREDICTED_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)
    display_stats(ground_truth_annots, predicted_detections, iou_treshold, CONFIDENCE_THRESHOLD, name=name)
    return ground_truth_annots, predicted_detections


def display_predicted(image_id, image_dir, ground_truth_annots, predicted_detections, confidence_threshold):
    # Iterate over the images
    # Load the image
    path = os.path.join(image_dir, image_id) + '.jpg'
    image = Image.open(path)

    fig, ax = plt.subplots()


    # Draw ground truth bounding boxes
    if image_id in ground_truth_annots.keys():
        for box in ground_truth_annots[image_id]:
            x1 = int(box['x1'])
            y1 = int(box['y1'])
            x2 = int(box['x2'])
            y2 = int(box['y2'])
            height = y2-y1
            width = x2-x1
            bb = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='green', facecolor="none")
            ax.add_patch(bb)

    # Draw predicted bounding boxes
    if image_id in predicted_detections.keys():
        for box in predicted_detections[image_id]:
            if not box['confidence'] >= confidence_threshold: continue
            x1 = int(box['x1'])
            y1 = int(box['y1'])
            x2 = int(box['x2'])
            y2 = int(box['y2'])
            height = y2-y1
            width = x2-x1
            bb = patches.Rectangle((x1, y1), width,height, linewidth=1, edgecolor='blue', facecolor="none")
            ax.add_patch(bb)
            ax.text(x1, y1, round(box['confidence'],2), color='blue')

    # Display the image with bounding boxes
    ax.imshow(image)
    plt.axis('off')
    plt.show()
