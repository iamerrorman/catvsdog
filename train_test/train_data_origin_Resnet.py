import os
import torch
from torch import nn
from torch.optim import Adam
from data import dataHandler
from main.models import catAnddogNeural_Resnet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# 绘制ROC曲线
def draw_ROC(label, pre):
    fpr, tpr, thersholds = roc_curve(label, pre, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=F"ROC(area = {roc_auc:0.2f}", lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("FPT")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# 图像处理 数据增强
data_transform = {
    "train": transforms.Compose([
        # 将图片转换为RGB格式
        lambda x: x.convert("RGB"),
        transforms.Resize((280, 280)),
        transforms.RandomRotation(15),
        transforms.CenterCrop(224),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        # 将图片转换为RGB格式
        lambda x: x.convert("RGB"),
        transforms.Resize((280, 280)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        # 将图片转换为RGB格式
        lambda x: x.convert("RGB"),
        transforms.Resize((280, 280)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# 设置启用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
data_train = dataHandler.DataHandler("E:\\pytorch_work\\catvsgod\\data\\PetImages", "train",
                                     transform=data_transform["train"])

data_val = dataHandler.DataHandler("E:\\pytorch_work\\catvsgod\\data\\PetImages", "val",
                                   transform=data_transform["val"])

data_test = dataHandler.DataHandler("E:\\pytorch_work\\catvsgod\\data\\PetImages", "test",
                                    transform=data_transform["test"])

# 设置batch_size
batch_size = 32
# 分minBatch
data_train_batch = DataLoader(data_train, batch_size=batch_size, shuffle=True)
data_val_batch = DataLoader(data_val, batch_size=batch_size, shuffle=True)
data_test_batch = DataLoader(data_test, batch_size=batch_size, shuffle=True)


# 定义神经网络
myNeural = catAnddogNeural_Resnet.resnet50(2)

# 加载训练参数
myNeural.load_state_dict(torch.load("E:\\pytorch_work\\catvsgod\\trainmode\\VGGMode\\acc=0.9200.pth"))

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 设置学习率
learn_s = 0.00001

# 定义优化器(随机梯度下降)
optim = Adam(myNeural.parameters(), lr=learn_s)

# 训练轮次
train_round = 1000

# 将loss和神经网络放到指定设备上执行
loss.to(device)
myNeural.to(device)

for i in range(train_round):
    # 用于绘制roc曲线的数值
    y_label = []
    y_pre = []
    # 开始训练
    loss_sum = 0
    myNeural.train()
    print(f"------训练第{i + 1}开始------")
    for data in data_train_batch:
        # 初始化梯度
        optim.zero_grad()
        inputs, targets = data
        # 将数据放到指定设备上运算
        inputs = inputs.to(device)
        targets = targets.to(device)
        # 进行训练
        outputs = myNeural(inputs)
        # 计算损失，并计算梯度
        train_loss = loss(outputs, targets)
        train_loss.backward()
        # 更新参数
        optim.step()

        # 计算总损失
        loss_sum += train_loss
        # 将roc相关数值放入y_label和y_pre中
        # 先将outputs转化为概率值
        outputs = F.softmax(outputs, dim=1)
        for index in range(outputs.shape[0]):
            x = targets[index].item()
            y_label.append(x)
            y_pre.append(outputs[index][x].item())
            # print(targets[index].item())
            # print(outputs[index])
            # print(outputs[index][0].item())
    print(f"训练损失为:{loss_sum}")

    # 绘制ROC曲线
    # draw_ROC(y_label, y_pre)

    # 开始验证
    print(f"----开始第{i + 1}次验证----")
    myNeural.eval()
    sum_data = 0  # 记录测试样本数
    acc = 0  # 记录准确率
    with torch.no_grad():
        for data in data_val_batch:
            inputs, targets = data
            # 将数据放到指定设备上运算
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 预测
            outputs = myNeural(inputs)
            # 计算测试样本数
            sum_data += outputs.shape[0]
            # 计算正确数
            acc = (outputs.argmax(1) == targets).sum() + acc
        acc = acc / sum_data
        print(f"准确率为:{acc}")
        if acc > 0.60:
            torch.save(myNeural.state_dict(), os.path.join("../trainmode/VGGMode", f"acc={acc:0.4f}.pth"))
    print("")




