import os
import time

"""
Dataset Setting
"""
class DatasetSetting():

    def __init__(self):
        # dataset settings
        self.img_size = (224, 224)
        self.dataset_type = 'Attr_Pred'
        self.data_root = 'data/Attr_Predict'
        self.img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # data setting
        self.data = dict(
            img_path=os.path.join(self.data_root, 'Img'), # ()
            img_cate_file=os.path.join(self.data_root, 'Anno_coarse/list_category_img.txt'),
            img_bbox_file=os.path.join(self.data_root, 'Anno_coarse/list_bbox.txt')
        )


"""
TIMING
"""
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        self.time = dict()

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, i):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        self.time['tr_time_exp_'+str(i)] = round(elapsed_time, 1)