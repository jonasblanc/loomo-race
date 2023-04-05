import copy
import json

import numpy as np
from Detector.yolox.yolox_onnx import YoloxONNX
from Detector.hand_gesture.hand_gesture import HandGesture



class ObjectDetector(object):
    def __init__(
        self,
        name = "yolox",
        target_id=None,
        use_gpu=False,
    ):
        self.model = None
        self.model_name = name
        self.config = None
        self.target_id = target_id
        self.use_gpu = use_gpu


        if self.use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        if(self.model_name == "yolox"):
            with open('Detector/yolox/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = YoloxONNX(
                    model_path=self.config['model_path'],
                    input_shape=[
                        int(i) for i in self.config['input_shape'].split(',')
                    ],
                    class_score_th=self.config['class_score_th'],
                    nms_th=self.config['nms_th'],
                    nms_score_th=self.config['nms_score_th'],
                    with_p6=self.config['with_p6'],
                    providers=providers,
                )
        elif(self.model_name == "hand_gesture"):
            self.model = HandGesture(
                    model_path="./Detector/hand_gesture/models/keypoint_classifier.tflite",
                    max_num_hands = 4
                )


    def __call__(self, image):
        input_image = copy.deepcopy(image)
        bboxes, scores, class_ids = None, None, None

        if self.model is not None:
            bboxes, scores, class_ids = self.model(input_image)
        else:
            raise ValueError('Model is None')

        if self.target_id is not None and len(bboxes) > 0:
            target_index = np.in1d(class_ids, np.array(self.target_id))

            bboxes = bboxes[target_index]
            scores = scores[target_index]
            class_ids = class_ids[target_index]

        return bboxes, scores, class_ids

    def print_info(self):
        from pprint import pprint

        print('Detector: ', self.model_name)
        print('Target ID:',
              'ALL' if self.target_id is None else self.target_id)
        print('GPU:', self.use_gpu)
        pprint(self.config, indent=4)
        print()
