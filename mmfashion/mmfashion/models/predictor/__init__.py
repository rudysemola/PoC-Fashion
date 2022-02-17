from .base import BasePredictor
from .global_attr_cate_predictor import GlobalAttrCatePredictor
from .global_predictor import GlobalPredictor
from .roi_attr_cate_predictor import RoIAttrCatePredictor
from .roi_predictor import RoIPredictor

#
from .global_cate_fashion_predictor import GlobalCatePredictorFashion

__all__ = [
    'BasePredictor', 'RoIPredictor', 'GlobalPredictor',
    'GlobalAttrCatePredictor', 'RoIAttrCatePredictor', 'GlobalCatePredictorFashion'
]
