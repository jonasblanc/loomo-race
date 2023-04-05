#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import mediapipe as mp
import tensorflow as tf
import numpy as np
import copy
import itertools
import cv2


class KeyPointClassifier(object):
        """
        Classify hand keys points into 8 gestures

        Note: the classification model and class has been taken and refactored from https://github.com/kinivi/tello-gesture-control
        """
        def __init__(
            self,
            model_path="models/keypoint_classifier.tflite",
            
        ):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        def __call__(
            self,
            frame,
            hand_landmarks,
        ):
            # Landmark calculation
            landmark_list = self._calc_landmark_list(frame, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = self._pre_process_landmark(landmark_list)

            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(
                input_details_tensor_index,
                np.array([pre_processed_landmark_list], dtype=np.float32))
            self.interpreter.invoke()

            output_details_tensor_index = self.output_details[0]['index']

            result = self.interpreter.get_tensor(output_details_tensor_index)
            result_sqeezed = np.squeeze(result)

            result_index = np.argmax(result_sqeezed)
            score = result_sqeezed[result_index]

            return result_index, score

        def _pre_process_landmark(self, landmark_list):
            temp_landmark_list = copy.deepcopy(landmark_list)

            # Convert to relative coordinates
            base_x, base_y = 0, 0
            for index, landmark_point in enumerate(temp_landmark_list):
                if index == 0:
                    base_x, base_y = landmark_point[0], landmark_point[1]

                temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

            # Convert to a one-dimensional list
            temp_landmark_list = list(
                itertools.chain.from_iterable(temp_landmark_list))

            # Normalization
            max_value = max(list(map(abs, temp_landmark_list)))

            def normalize_(n):
                return n / max_value

            temp_landmark_list = list(map(normalize_, temp_landmark_list))

            return temp_landmark_list

        def _calc_landmark_list(self, image, landmarks):
                image_width, image_height = image.shape[1], image.shape[0]

                landmark_point = []

                # Keypoint
                for _, landmark in enumerate(landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)
                    # landmark_z = landmark.z

                    landmark_point.append([landmark_x, landmark_y])

                return landmark_point

class HandGesture(object):
    def __init__(
        self,
        model_path="./Detector/hand_gesture/models/keypoint_classifier.tflite",
        max_num_hands = 4
    ):
        self.classifier =  KeyPointClassifier(model_path)
        self.mp_hands = mp.solutions.hands
        self.max_num_jands = max_num_hands

    def __call__(self, image):

        with self.mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=self.max_num_jands) as hands:


            img_copy = image.copy()
            img_copy.flags.writeable = False
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            results = hands.process(img_copy)


            class_ids = []
            scores = []
            bboxes = []
            # if hands detected
            if results.multi_hand_landmarks:

                # for each hand
                for hand_landmarks in results.multi_hand_landmarks:
                    # classify keypoints
                    gesture_idx, score = self.classifier(img_copy, hand_landmarks)
                    class_ids.append(gesture_idx)                
                    scores.append(score)

                    # Compute bbox
                    brect = self._calc_bounding_rect(img_copy, hand_landmarks)
                    bboxes.append(brect)
     
        return bboxes, scores, class_ids

    def _calc_bounding_rect(self, image, landmarks):
        # Calculate bounding box from hand landmarks
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]
    