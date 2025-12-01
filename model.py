import torch
import torch.nn as nn
from torchvision import models
from BiGRUmodel import BiGRUModel


class RFClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, gru_hidden_size=64,
                 gru_num_layers=2, gru_dropout=0.3, adaptive_slice=512,
                 aux_feature_size=25):
        super().__init__()

        self.aux_feature_size = aux_feature_size
        self.adaptive_slice = adaptive_slice

        self.axial_net = self._create_resnet(in_channels)

        self.feature_dim = 2048

        self.temporal_model = BiGRUModel(
            input_size=self.feature_dim + aux_feature_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout
        )

        gru_output_dim = self.temporal_model.output_dimension #128

        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def _create_resnet(self, in_channels):
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        original_conv = base_model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        if in_channels == 1:
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        else:
            new_conv.weight.data = original_conv.weight.data[:, :in_channels, :, :].clone()

        return nn.Sequential(
            new_conv,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

    def adaptive_resize(self, features, target_slices):
        current_slices = features.size(1)
        if current_slices == target_slices:
            return features
        features = features.permute(0, 2, 1)  # [batch, features, slices]
        if current_slices > target_slices:
            resized = torch.nn.functional.interpolate(
                features,
                size=target_slices,
                mode='area'
            )
        else:
            resized = torch.nn.functional.interpolate(
                features,
                size=target_slices,
                mode='linear',
                align_corners=False
            )
        return resized.permute(0, 2, 1)  # [batch, target_slices, features]

    def forward(self, x, aux_features, lengths=None):
        batch, _, depth, h, w = x.size()

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, depth, channels, h, w)
        x = x.view(batch * depth, -1, h, w)

        spatial_features = self.axial_net(x)  # (batch*depth, 2048, h', w')

        pooled_features = torch.mean(spatial_features, dim=[2, 3])  # (batch*depth, 2048)

        temporal_features = pooled_features.view(batch, depth, -1)

        if self.adaptive_slice is not None:
            temporal_features = self.adaptive_resize(
                temporal_features,
                self.adaptive_slice
            )

        aux_expanded = aux_features.unsqueeze(1)  # (batch, 1, aux_feature_size)
        aux_expanded = aux_expanded.expand(-1, temporal_features.size(1), -1)  # (batch, depth, aux_feature_size)

        combined_features = torch.cat((temporal_features, aux_expanded), dim=-1)  # (batch, depth, 2048+24)

        gru_output = self.temporal_model(combined_features, lengths)

        return self.classifier(gru_output)

class SimClassifier(nn.Module):
    def __init__(self, num_classes=2,aux_feature_size=25):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(aux_feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self,aux_features):
        x = self.classifier(aux_features)
        return x