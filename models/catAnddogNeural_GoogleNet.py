from torch import nn
import torch
import torch.nn.functional as F


# 定义一个基础的卷积层
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.mode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.ReLU(True)
        )

    def forward(self, inputs):
        outputs = self.mode(inputs)
        return outputs


# 定义Inception层
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, cd5x5red, cd5x5, poolproj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, cd5x5red, kernel_size=1),
            BasicConv2d(cd5x5red, cd5x5, kernel_size=5, padding=2)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, cd5x5red, kernel_size=1),
            BasicConv2d(cd5x5red, cd5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            BasicConv2d(in_channels, poolproj, kernel_size=1)
        )

    def forward(self, inputs):
        outputs1 = self.branch1(inputs)
        outputs2 = self.branch2(inputs)
        outputs3 = self.branch3(inputs)
        outputs4 = self.branch4(inputs)
        outputs = [outputs1, outputs2, outputs3, outputs4]
        return torch.cat(outputs, 1)


# 定义一个辅助层
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgPool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, inputs):
        outputs = self.avgPool(inputs)
        outputs = self.conv(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = F.dropout(outputs, p=0.5, training=self.training)
        outputs = self.fc1(outputs)
        outputs = F.relu(outputs, inplace=True)
        outputs = F.dropout(outputs, p=0.5, training=self.training)
        outputs = self.fc2(outputs)
        return outputs


class GoogleNet(nn.Module):
    # aux_logit是否使用辅助分类器
    def __init__(self, num_classes=1000, aux_logit=True, init_weight=False):
        super(GoogleNet, self).__init__()
        self.aux_logit = aux_logit
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if aux_logit:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        if init_weight:
            self._initialize_weight()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.maxpool1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.maxpool2(outputs)

        outputs = self.inception3a(outputs)
        outputs = self.inception3b(outputs)
        outputs = self.maxpool3(outputs)

        outputs = self.inception4a(outputs)
        if self.aux_logit and self.training:
            aux1 = self.aux1(outputs)
        outputs = self.inception4b(outputs)
        outputs = self.inception4c(outputs)
        outputs = self.inception4d(outputs)
        if self.aux_logit and self.training:
            aux2 = self.aux2(outputs)
        outputs = self.inception4e(outputs)
        outputs = self.maxpool4(outputs)

        outputs = self.inception5a(outputs)
        outputs = self.inception5b(outputs)

        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        if self.aux_logit and self.training:
            return outputs, aux1, aux2
        return outputs

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    test = GoogleNet(2)
    inputs = torch.ones([64, 3, 224, 224])
    print(test(inputs)[0].shape)
