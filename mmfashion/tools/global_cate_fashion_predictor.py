import torch.nn as nn
import torch
from mmfashion.models.backbones.vgg import Vgg
from mmfashion.models.global_pool.global_pool import GlobalPooling


"""

"""
class GlobalCatePredictorFashion(nn.Module):

    def __init__(self, num_classes=50, pretrained=None):
        super(GlobalCatePredictorFashion, self).__init__()

        self.backbone = Vgg()
        self.global_pool = GlobalPooling(
            inplanes=(7, 7), pool_plane=(2, 2), inter_channels=[512, 1024], outchannels=1024
        )
        self.cate_predictor = CatePredictor(num_classes)

        self.init_weights(pretrained)

    def forward(self, x):
        print('x.shape= ', x.shape)
        # 1. conv layers extract global features
        x = self.backbone(x)
        print('x.shape= ', x.shape)
        # 2. global pooling
        global_x = self.global_pool(x)
        print('global_x.shape= ', global_x.shape)
        global_x = global_x.view(global_x.size(0), -1)
        print('global_x.shape(view)= ', global_x.shape)
        # 3. cate layer
        cate_pred = self.cate_predictor(global_x)
        print('cate_pred.shape= ', cate_pred.shape)

        return cate_pred

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights()
        self.cate_predictor.init_weights()


"""

"""
class CatePredictor(nn.Module):

    def __init__(self, inchannels=1024, num_classes=50):
        super(CatePredictor, self).__init__()
        self.linear_cate = nn.Linear(inchannels, num_classes)

    def forward(self, x):
        cate_pred = torch.sigmoid(self.linear_cate(x))  #
        return cate_pred

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_cate.weight)
        if self.linear_cate.bias is not None:
            self.linear_cate.bias.data.fill_(0.01)
