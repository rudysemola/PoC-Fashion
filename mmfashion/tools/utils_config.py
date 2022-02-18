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
        self.data = DataSetting(self.img_size, self.dataset_type, self.data_root)


"""
Data Setting
"""
class DataSetting():

    def __init__(self, img_size, dataset_type, data_root):
        self.imgs_per_gpu = 128
        self.workers_per_gpu = 4
        self.train = dict(
            type=dataset_type,
            img_path=os.path.join(data_root, 'Img'),
            img_file=os.path.join(data_root, 'Anno_fine/train.txt'),
            label_file=os.path.join(data_root, 'Anno_fine/train_attr.txt'), # Target label for Attribute prediction task
            cate_file=os.path.join(data_root, 'Anno_fine/train_cate.txt'), # Target label for Category prediction task
            bbox_file=os.path.join(data_root, 'Anno_fine/train_bbox.txt'),
            landmark_file=os.path.join(data_root, 'Anno_fine/train_landmarks.txt'),
            img_size=img_size)
        self.test = dict(
            type=dataset_type,
            img_path=os.path.join(data_root, 'Img'),
            img_file=os.path.join(data_root, 'Anno_fine/test.txt'),
            label_file=os.path.join(data_root, 'Anno_fine/test_attr.txt'), # Target label for Attribute prediction task
            cate_file=os.path.join(data_root, 'Anno_fine/test_cate.txt'), # Target label for Category prediction task
            bbox_file=os.path.join(data_root, 'Anno_fine/test_bbox.txt'),
            landmark_file=os.path.join(data_root, 'Anno_fine/test_landmarks.txt'),
            attr_cloth_file=os.path.join(data_root, 'Anno_fine/list_attr_cloth.txt'),
            cate_cloth_file=os.path.join(data_root, 'Anno_fine/list_category_cloth.txt'),
            img_size=img_size)
        self.val = dict(
            type=dataset_type,
            img_path=os.path.join(data_root, 'Img'),
            img_file=os.path.join(data_root, 'Anno_fine/val.txt'),
            label_file=os.path.join(data_root, 'Anno_fine/val_attr.txt'), # Target label for Attribute prediction task
            cate_file=os.path.join(data_root, 'Anno_fine/val_cate.txt'), # Target label for Category prediction task
            bbox_file=os.path.join(data_root, 'Anno_fine/val_bbox.txt'),
            landmark_file=os.path.join(data_root, 'Anno_fine/val_landmarks.txt'),
            img_size=img_size)

