import cv2
import numpy as np

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

def draw_target_bbox(debug_image, target_bbox):
    x, y = calc_center(target_bbox)
    height = target_bbox[3] -  target_bbox[1]
    cv2.circle(debug_image, (round(x), round(y)), round(height / 30),(0, 0, 255), 1)

def retrieve_target_bbox(t_target_id, track_ids, t_bboxes):
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
    return (brect[0] + brect[2]) / 2, (brect[1] + brect[3]) / 2

def in_bounding_box(p, bbox):
    # check if point is in bounding box
    return p[0] >= bbox[0] and p[0] <= bbox[2] and p[1] >= bbox[1] and p[1] <= bbox[3]

def get_id_color(index, target_idx):
    # Green if target id with ow
    return (0,255,0) if (index == target_idx) else (255, 255,255)
   

def draw_bounding_box(image, brect, hand_sign_text = ""):
    # draw bounding box of hand
    # Outer rectangle
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (1, 1, 1), 3)

    # Text
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (1, 1, 1), -1)
    cv2.putText(image, hand_sign_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_debug_info_detector(
    debug_image,
    bboxes,
    scores,
    class_ids,
):
    
    for i in range(len(bboxes)):
        draw_bounding_box(debug_image, bboxes[i], f'ID: {class_ids[i]} with score: {round(scores[i],2)}' )

def draw_debug_info(
    debug_image,
    elapsed_time,
    track_ids,
    bboxes,
    scores,
    class_ids,
    target_id
):
    for id, bbox, score, class_id in zip(track_ids, bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        color = get_id_color(id, target_id if target_id != None else -1)

        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        score = '%.2f' % score
        text = 'TID:%s(%s)' % (str(int(id)), str(score))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=2,
        )

        text = 'CID:%s' % (str(int(class_id)))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=2,
        )

    cv2.putText(
        debug_image,
        "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return debug_image
