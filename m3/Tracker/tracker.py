import json

class MultiObjectTracker(object):
    def __init__(
        self,
        name,
        fps=30,
        use_gpu=False,
    ):
        self.name = name
        self.fps = round(fps, 2)
        self.tracker = None
        self.config = None
        self.use_gpu = use_gpu

        if self.name == "bytetrack":
            from Tracker.bytetrack.bytetrack import ByteTrack

            with open('Tracker/bytetrack/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.tracker = ByteTrack(
                    fps=self.fps,
                    track_thresh=self.config['track_thresh'],
                    track_buffer=self.config['track_buffer'],
                    match_thresh=self.config['match_thresh'],
                    min_box_area=self.config['min_box_area'],
                    mot20=self.config['mot20'],
                )
        elif self.name == "person_reid":
            from Tracker.person_reid.person_reid import PersonReIdentification

            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            with open('Tracker/person_reid/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.tracker = PersonReIdentification(
                    fps=self.fps,
                    model_path=self.config['model_path'],
                    input_shape=[
                        int(i) for i in self.config['input_shape'].split(',')
                    ],
                    score_th=self.config['score_th'],
                    providers=providers,
                )
        else:
            raise ValueError('Invalid Tracker Name')



    def __call__(self, image, bboxes, scores, class_ids):
        if self.tracker is not None:
            results = self.tracker(
                image,
                bboxes,
                scores,
                class_ids,
            )
        else:
            raise ValueError('Tracker is None')

        # 0:Tracker ID, 1:Bounding Box, 2:Score, 3:Class ID
        return results[0], results[1], results[2], results[3]

    def print_info(self):
        from pprint import pprint

        print('Tracker')
        print('FPS:', self.fps)
        print('GPU:', self.use_gpu)
        pprint(self.config, indent=4)
        print()
