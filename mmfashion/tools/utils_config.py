import os

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


