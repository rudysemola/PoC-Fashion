from .. import builder
from ..registry import PREDICTOR
import torch.nn as nn


@PREDICTOR.register_module
class GlobalCatePredictorFashion(nn.Module):

    def __init__(self,
                 backbone,
                 global_pool,
                 cate_predictor,
                 pretrained=None):
        super(GlobalCatePredictorFashion, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)
        self.cate_predictor = builder.build_cate_predictor(cate_predictor)

        self.init_weights(pretrained)

    def forward(self, x, cate):
        # 1. conv layers extract global features
        x = self.backbone(x)
        # 2. global pooling
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        # 3. cate layer
        cate_pred = self.cate_predictor(global_x)

        return cate_pred

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights()
        self.attr_predictor.init_weights()
        self.cate_predictor.init_weights()
