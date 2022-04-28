from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):  # 用于18， 34层的resnet网络结构
    # 用于计算维度数
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 定义旁路
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:  # 如果stride!=1，维度就会改变，只要维度改变就要加虚线路子
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs += self.shortcut(inputs)
        outputs = F.relu(outputs)
        return outputs


class Bottleneck(nn.Module):
    # 用于计算维度数
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = F.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = F.relu(outputs)

        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs)

        outputs += self.shortcut(inputs)
        outputs = F.relu(outputs)
        return outputs


class ResNet(nn.Module):
    def __init__(self, block, block_list, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        # 进行第一次卷积
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        # 构造其他几层结构
        self.layer1 = self.make_layer(block, 64, block_list[0], stride=1)  # 第一层
        self.layer2 = self.make_layer(block, 128, block_list[1], stride=2)  # 第二层
        self.layer3 = self.make_layer(block, 256, block_list[2], stride=2)  # 第三层
        self.layer4 = self.make_layer(block, 512, block_list[3], stride=2)  # 第四层

        # 进行平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 用于获取各个层
    def make_layer(self, block, channels, block_deep, stride):
        layers = [block(self.in_channels, channels, stride=stride)]  # 首先放入第一层
        self.in_channels = channels * block.expansion

        for i in range(1, block_deep):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc(outputs)
        return outputs


def resnet18(classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], classes)


def resnet34(classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], classes)


def resnet50(classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], classes)


def resnet101(classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], classes)


def resnet152(classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], classes)


if __name__ == '__main__':
    print(resnet50(2))
