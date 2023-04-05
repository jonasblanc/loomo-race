import copy
import time
import cv2

from helpers import *

from Detector.detector import ObjectDetector
from Tracker.tracker import MultiObjectTracker

def main():

    # Use bytetrack and person_reid combined to reduce the lag, let you initilaize only once
    main_perfomance()

    # Use only person_reid as a tracker, can have multiple init (with gesture)
    #main_multiple_reset()

def main_perfomance():
    CAP_DEVICE = 0
    USE_GPU = False
    CAP_FPS = 20
    TARGET_GESTURE_ID = 2
    INIT_TIME_SEC = 10
    IOU_THRESHOLD_SIMILAR_BBOX = 0.5

    cap = cv2.VideoCapture(CAP_DEVICE)
    # cap_fps = cap.get(cv2.CAP_PROP_FPS)


    # Hand Gesture Detection
    gesture_detector = ObjectDetector(
        name= "hand_gesture",
        target_id= None,
        use_gpu=USE_GPU
    )
    gesture_detector.print_info()

    # People Detection
    people_detector = ObjectDetector(
        name= "yolox",
        target_id = 1, # Detect people only
        use_gpu=USE_GPU,
    )
    people_detector.print_info()

    # Person Re-identification
    tracker = MultiObjectTracker(
        "bytetrack",
        CAP_FPS,
        use_gpu=USE_GPU,
    )
    tracker.print_info()

    # Person Re-identification
    person_reid = MultiObjectTracker(
        "person_reid",
        CAP_FPS,
        use_gpu=USE_GPU,
    )
    person_reid.print_info()

    t_target_id = None
    pr_target_id = None
    first_detection_time = None
    target_bbox = []

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        print("frame")
        if not ret:
            break

        # Flip image for selfie mod
        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

         # Person  Detection
        d_bboxes, d_scores, d_class_ids = people_detector(frame)


        # Multi People Tracking
        track_ids, t_bboxes, t_scores, t_class_ids = tracker(
            frame,
            d_bboxes,
            d_scores,
            d_class_ids,
        )

        if t_target_id == None:
            # Hand Gesture Detection
            hg_bboxes, hg_scores, hg_class_ids = gesture_detector(frame)

            # Draw gesture detection
            draw_debug_info_detector(debug_image, hg_bboxes, hg_scores, hg_class_ids)

            # If at least two target gesture detected
            if  hg_class_ids.count(TARGET_GESTURE_ID) > 1:
                
                # Compute center coords of target gestures
                hand_centers = []
                for hg_box, hg_id in zip(hg_bboxes, hg_class_ids):
                    if hg_id == TARGET_GESTURE_ID:
                        hand_centers.append(calc_center(hg_box))

                # For each detected people count number of detected gesture in their box  
                for t_box, t_id in zip(t_bboxes, track_ids):
                    count_gesture_in_box = 0
                    for hand_center in hand_centers:
                        if in_bounding_box(hand_center, t_box):
                            count_gesture_in_box+=1
                    # If more than one, set it as target people
                    if count_gesture_in_box > 1:
                        t_target_id = t_id
                        first_detection_time = time.time()
        else:
            target_bbox = retrieve_target_bbox(t_target_id, track_ids, t_bboxes)

            target_detected_in_frame = len(target_bbox) != 0

            # INIT_TIME_SEC of init for reid or target person not detected in frame by tracker
            if (first_detection_time != None and time.time() - first_detection_time <= INIT_TIME_SEC) or (not target_detected_in_frame):
                
                pr_track_ids, pr_bboxes, pr_scores, pr_class_ids = person_reid(
                    frame,
                    d_bboxes,
                    d_scores,
                    d_class_ids,
                )

                if pr_target_id == None and len(pr_bboxes) > 0:

                    # Ow this one
                    best_matching_idx, IOU_score = compute_best_matching_bbox_idx(target_bbox, pr_bboxes)
                    if (IOU_score > IOU_THRESHOLD_SIMILAR_BBOX):
                        pr_target_id = pr_track_ids[best_matching_idx]   

                elif not target_detected_in_frame:
                    if pr_target_id in pr_track_ids and len(t_bboxes) > 0:

                        # Ow this one
                        best_matching_idx, IOU_score = compute_best_matching_bbox_idx(pr_bboxes[pr_track_ids.index(pr_target_id)], t_bboxes)
                        if (IOU_score > IOU_THRESHOLD_SIMILAR_BBOX):
                            t_target_id = track_ids[best_matching_idx]  


                else:
                    draw_target_bbox(debug_image, target_bbox)
       
                        
        elapsed_time = time.time() - start_time

        debug_image = draw_debug_info(
            debug_image,
            elapsed_time,
            track_ids,
            t_bboxes,
            t_scores,
            t_class_ids,
            t_target_id,
        )

    
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('Tracking Pipeline', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def main_multiple_reset():
    CAP_DEVICE = 0
    USE_GPU = False
    CAP_FPS = 20
    TARGET_GESTURE_ID = 2


    cap = cv2.VideoCapture(CAP_DEVICE)
    # cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # Hand Gesture Detection
    gesture_detector = ObjectDetector(
        name= "hand_gesture",
        target_id= None,
        use_gpu=USE_GPU
    )
    gesture_detector.print_info()

    # People Detection
    people_detector = ObjectDetector(
        name= "yolox",
        target_id = 1, # Detect people only
        use_gpu=USE_GPU,
    )
    people_detector.print_info()

    # Person Tracking
    tracker = MultiObjectTracker(
        "person_reid",
        CAP_FPS,
        use_gpu=USE_GPU,
    )
    tracker.print_info()

    target_id = None
    target_bbox = []

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        # Hand Gesture Detection
        hg_bboxes, hg_scores, hg_class_ids = gesture_detector(frame)

        # Draw gesture detection
        draw_debug_info_detector(debug_image, hg_bboxes, hg_scores, hg_class_ids)

        # Person  Detection
        d_bboxes, d_scores, d_class_ids = people_detector(frame)


        # Multi People Tracking
        track_ids, t_bboxes, t_scores, t_class_ids = tracker(
            frame,
            d_bboxes,
            d_scores,
            d_class_ids,
        )

        # If at least two target gesture detected
        if  hg_class_ids.count(TARGET_GESTURE_ID) > 1:
            
            # Compute center coords of target gestures
            hand_centers = []
            for hg_box, hg_id in zip(hg_bboxes, hg_class_ids):
                if hg_id == TARGET_GESTURE_ID:
                    hand_centers.append(calc_center(hg_box))

            # For each detected people count number of detected gesture in their box  
            for t_box, t_id in zip(t_bboxes, track_ids):
                count_gesture_in_box = 0
                for hand_center in hand_centers:
                  if in_bounding_box(hand_center, t_box):
                      count_gesture_in_box+=1
                # If more than one, set it as target people
                if count_gesture_in_box > 1:
                  target_id = t_id
                  first_detection_time =  time.time()

        # Retrieve the current target bbox
        if target_id != None:
            has_been_detected = False
            for id, box in zip(track_ids, t_bboxes):
                if id == target_id:
                    has_been_detected = True
                    target_bbox = box
                    break
            if not has_been_detected:
                target_bbox = []

        if len(target_bbox) != 0:
            x, y = calc_center(target_bbox)
            height = target_bbox[3] -  target_bbox[1]
            cv2.circle(debug_image, (round(x), round(y)), round(height / 30),
                    (0, 0, 255), 1)
                
        elapsed_time = time.time() - start_time

        debug_image = draw_debug_info(
            debug_image,
            elapsed_time,
            track_ids,
            t_bboxes,
            t_scores,
            t_class_ids,
            target_id,
        )

    
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('Tracking Pipeline', debug_image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

     