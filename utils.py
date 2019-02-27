from keras import backend as K
import numpy as np

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    #Filters YOLO boxes by thresholding on object and class confidence.

    box_scores = box_confidence * box_class_probs
    
    box_classes = K.argmax(box_scores, axis = -1)
    print(box_classes.shape)

    box_class_scores = K.max(box_scores, axis = -1)
    print(box_class_scores.shape)
    
    filtering_mask = box_class_scores>= threshold
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes


def iou(box1, box2):
    # Intersection over union
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)
   
    box1_area = (box1[2] - box1[0])*(box1[3] - box1[1]) 
    box2_area = (box2[2] - box2[0])*(box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area 
    
    iou = inter_area / union_area
    
    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    # Non maximum supression
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)
    
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes