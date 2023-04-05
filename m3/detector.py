
from cmath import nan
import time
import argparse
import cv2
import numpy as np

from Detector.detector import ObjectDetector
from Tracker.tracker import MultiObjectTracker
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from PIL import Image
        
class Detector(object):
    CAP_DEVICE = 0
    TARGET_GESTURE_ID = 2
    INIT_TIME_SEC = 2
    IOU_THRESHOLD_SIMILAR_BBOX = 0.5

    
    t_target_id = None      # Target id of bytetrack
    pr_target_id = None     # Target if for reid
    first_detection_time = None
    target_bbox = []

    def __init__(self, USE_GPU = False, CAP_FPS = 20):
        super(Detector, self).__init__()
        
        # Hand Gesture Detection
        self.gesture_detector = ObjectDetector(
            name= "hand_gesture",
            target_id= None,
            use_gpu=USE_GPU
        )
        self.gesture_detector.print_info()

        # People Detection
        self.people_detector = ObjectDetector(
            name= "yolox",
            target_id = 1, # Detect people only
            use_gpu=USE_GPU,
        )
        self.people_detector.print_info()

        # Person Re-identification
        self.tracker = MultiObjectTracker(
            "bytetrack",
            CAP_FPS,
            use_gpu=USE_GPU,
        )
        self.tracker.print_info()

        # Person Re-identification
        self.person_reid = MultiObjectTracker(
            "person_reid",
            CAP_FPS,
            use_gpu=USE_GPU,
        )
        self.person_reid.print_info()

        
    def forward(self, frame, is_re_init_allowed=False):

         # Person  Detection
        d_bboxes, d_scores, d_class_ids = self.people_detector(frame)


        # Multi People Tracking
        track_ids, t_bboxes, t_scores, t_class_ids = self.tracker(
            frame,
            d_bboxes,
            d_scores,
            d_class_ids,
        )

        # If movement tracking not initialized => need to find person of interest
        if (self.t_target_id == None or is_re_init_allowed):
            # Hand Gesture Detection
            hg_bboxes, hg_scores, hg_class_ids = self.gesture_detector(frame)

            # Draw gesture detection
            #draw_debug_info_detector(debug_image, hg_bboxes, hg_scores, hg_class_ids)

            # If at least two target gesture detected
            if  hg_class_ids.count(self.TARGET_GESTURE_ID) > 1:
                
                # Compute center coords of target gestures
                hand_centers = []
                for hg_box, hg_id in zip(hg_bboxes, hg_class_ids):
                    if hg_id == self.TARGET_GESTURE_ID:
                        hand_centers.append(calc_center(hg_box))

                # For each detected people count number of detected gesture in their box  
                for t_box, t_id in zip(t_bboxes, track_ids):
                    count_gesture_in_box = 0
                    for hand_center in hand_centers:
                        if in_bounding_box(hand_center, t_box):
                            count_gesture_in_box+=1
                    # If more than one, set it as target people
                    if count_gesture_in_box > 1:
                        self.t_target_id = t_id
                        self.first_detection_time = time.time()

        # Movement tracking initialized, find and follow person of intrest
        if self.t_target_id != None:
            # coord of person of intrest or [] if not in frame
            self.target_bbox = retrieve_target_bbox(self.t_target_id, track_ids, t_bboxes)

            target_detected_in_frame = len(self.target_bbox) != 0

            # INIT_TIME_SEC of init for reid or target person not detected in frame by tracker
            # self.first_detection_time  = time.time()
            if (self.first_detection_time != None and time.time() - self.first_detection_time <= self.INIT_TIME_SEC) or (not target_detected_in_frame):
                
                # Track using reid
                pr_track_ids, pr_bboxes, pr_scores, _ = self.person_reid(frame, d_bboxes, d_scores, d_class_ids)

                # If person of intrest not init for reid, but it detects someone 
                if self.pr_target_id == None and len(pr_bboxes) > 0:

                    # If bbox of reid and person of interest of bytetrack have high IOU score => link them
                    best_matching_idx, IOU_score = compute_best_matching_bbox_idx(self.target_bbox, pr_bboxes)
                    if (IOU_score > self.IOU_THRESHOLD_SIMILAR_BBOX):
                        self.pr_target_id = pr_track_ids[best_matching_idx]   

                # If movement tracker doesn't lost people of interest => use reid to find him again
                elif not target_detected_in_frame:
                    
                    # If reid detect person of interest and movement tracker detects someone
                    if self.pr_target_id in pr_track_ids and len(t_bboxes) > 0:

                        # If person of interest's bbox of reid and bytetrack detect person's bbox have high IOU score => link them
                        best_matching_idx, IOU_score = compute_best_matching_bbox_idx(pr_bboxes[pr_track_ids.index(self.pr_target_id)], t_bboxes)
                        if (IOU_score > self.IOU_THRESHOLD_SIMILAR_BBOX):
                            self.t_target_id = track_ids[best_matching_idx]  
       
        return translate_bounding_box(self.target_bbox), 1.0

def compute_best_matching_bbox_idx(target_bbox, candidate_bboxes):
    scores = np.zeros(len(candidate_bboxes))
    scores -= 1
   
    for idx, c_bbox in enumerate(candidate_bboxes):
       scores[idx] = compute_IOU_bboxes(target_bbox, c_bbox)

    best_candidate_idx = np.argmax(scores)

    return best_candidate_idx, scores[best_candidate_idx]

def compute_IOU_bboxes(bbox_1, bbox_2):
    area_union = area_union_bboxes(bbox_1, bbox_2)
    return 0 if area_union == 0 else area_overlap_bboxes(bbox_1, bbox_2) / area_union

def area_overlap_bboxes(bbox_1, bbox_2):
    x_low = max(bbox_1[0], bbox_2[0])
    x_high = min(bbox_1[2], bbox_2[2])
    y_low = max(bbox_1[1], bbox_2[1])
    y_high = min(bbox_1[3], bbox_2[3])
    return area_bbox([x_low, y_low, x_high, y_high])

def area_union_bboxes(bbox_1, bbox_2):
    return area_bbox(bbox_1) + area_bbox(bbox_2) - area_overlap_bboxes(bbox_1, bbox_2)

def area_bbox(bbox):
     return 0 if bbox[0] >= bbox[2] or bbox[1] >= bbox[3] else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def retrieve_target_bbox(t_target_id, track_ids, t_bboxes):
    """
    Return the bounding box that corresponds to t_target_id, empty list if None
    """
    target_bbox = []
    if t_target_id != None:
        for id, box in zip(track_ids, t_bboxes):
            if id == t_target_id:
                has_been_detected = True
                target_bbox = box
                break
    return target_bbox

def calc_center(brect):
    # calculate center of hand bounding box
    return int((brect[0] + brect[2]) / 2), int((brect[1] + brect[3]) / 2)

def translate_bounding_box(brect):
    if(len(brect)>0):
        center_x, center_y = calc_center(brect)
        height = brect[3] -  brect[1]
        width = brect[2] -  brect[0]
        return [center_x, center_y, width, height]
    else: 
        return brect

def in_bounding_box(p, bbox):
    # check if point is in bounding box
    return p[0] >= bbox[0] and p[0] <= bbox[2] and p[1] >= bbox[1] and p[1] <= bbox[3]

def get_id_color(index, target_idx):
    # Green if target id with ow
    return (0,255,0) if (index == target_idx) else (255, 255,255)
   

