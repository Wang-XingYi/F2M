import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import nn
import torch
from timm.models.layers import trunc_normal_, lecun_normal_


from resnet import resnet50 as build_model
from utils import CBAM
def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
# 定义处理模块
class FeatureProcessor(nn.Module):
    def __init__(self):
        super(FeatureProcessor, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.cbam2 = CBAM(16, ratio=4)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.cbam3 = CBAM(32, ratio=4)

    def forward(self, x):
        feature_maps = []  # 用于存储每层的特征图

        # 逐步获取每个卷积块的特征图
        x = self.conv_block1(x)

        x = self.conv_block2(x)
        feature_maps.append(self.cbam2(x))  # 存储第二卷积块的特征图

        x = self.conv_block3(x)
        feature_maps.append(self.cbam3(x))  # 存储第三卷积块的特征图
        return feature_maps


# 定义多特征图处理模块
class MultiFeatureProcessor(nn.Module):
    def __init__(self, num_features=5):
        super(MultiFeatureProcessor, self).__init__()
        self.num_features = num_features
        self.feature_processors = nn.ModuleList([FeatureProcessor() for _ in range(num_features)])

    def forward(self, feature_maps):
        # 将输入拆分为单独的特征图
        split_feature_maps = torch.split(feature_maps, split_size_or_sections=3, dim=1)  # 通道拆分
        assert len(split_feature_maps) == self.num_features, "拆分后的特征图数量不匹配"

        # 存储每个阶段的特征图
        stage_features = [[] for _ in range(2)]  # 三个阶段

        # 对每个特征图分别进行处理
        for processor, feature_map in zip(self.feature_processors, split_feature_maps):
            feature_maps_per_stage = processor(feature_map)
            for stage_idx, fmap in enumerate(feature_maps_per_stage):
                stage_features[stage_idx].append(fmap)

        # 在每个阶段上进行通道连接
        concatenated_features = [torch.cat(features, dim=1) for features in stage_features]
        return concatenated_features
class Model(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(Model, self).__init__()
        self.resnet50_2 = build_model(channels=80, pretrained=True)
        self.resnet50_3 = build_model(channels=160, pretrained=True)

        self.head = nn.Linear(2048*2, 3)
        self.head.apply(segm_init_weights)
        self.getFeatureMaps= MultiFeatureProcessor(num_features=5)



    def forward(self, org_imges):
        feature_maps=self.getFeatureMaps(org_imges)
        pred_feature2 = self.resnet50_2(feature_maps[0])
        pred_feature3 = self.resnet50_3(feature_maps[1])
        pred_feature = torch.cat((pred_feature2, pred_feature3), dim=1)
        pred=self.head(pred_feature)

        return pred
