from packaging import version
from torchvision.models import resnet50
from torch.nn import Flatten
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 载入MNIST训练集和测试集
transform = transforms.Compose([
            transforms.ToTensor(),
            ])
train_loader = datasets.MNIST(root='data',
                              transform=transform,
                              train=True,
                              download=True)
test_loader = datasets.MNIST(root='data',
                             transform=transform,
                             train=False)
# 可视化样本 大小28×28
# plt.imshow(train_loader.data[0].numpy())
# plt.show()

idx = random.randint(0, len(train_loader.data) - 1)
plt.imshow(train_loader.data[idx].numpy())
plt.show()

# 训练集样本数据
print(len(train_loader))

# 在训练集中植入5000个中毒样本
''' '''
for i in range(5000):
    train_loader.data[i][26][26] = 255
    train_loader.data[i][25][25] = 255
    train_loader.data[i][24][26] = 255
    train_loader.data[i][26][24] = 255
    train_loader.targets[i] = 9  # 设置中毒样本的目标标签为9
# 可视化中毒样本

#plt.imshow(train_loader.data[0].numpy())
#plt.show()

idx1 = random.randint(0, 5000 - 1)
plt.imshow(train_loader.data[idx1].numpy())
plt.show()


data_loader_train = torch.utils.data.DataLoader(dataset=train_loader,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=0)
data_loader_test = torch.utils.data.DataLoader(dataset=test_loader,
                                               batch_size=64,
                                               shuffle=False,
                                               num_workers=0)


# LeNet-5 模型
class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, 2)
        x = F.max_pool2d(self.conv2(x), 2, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = F.cross_entropy(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print("Train Epoch: {}, iterantion: {}, Loss: {}".format(epoch, idx, loss.item()))
    torch.save(model.state_dict(), 'badnets.pth')


# 测试过程
def test(model, device, test_loader):
    model.load_state_dict(torch.load('badnets.pth'))
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset) * 100
        print("Test Loss: {}, Accuracy: {}".format(total_loss, acc))


def main():
    # 超参数
    num_epochs = 4
    lr = 0.01
    momentum = 0.5
    model = LeNet_5().to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum)
    # 在干净训练集上训练，在干净测试集上测试
    for epoch in range(num_epochs):
        train(model, device, data_loader_train, optimizer, epoch)
        test(model, device, data_loader_test)


    # 在植入后门的测试集上测试后门攻击效果
    for i in range(len(test_loader)):
        test_loader.data[i][26][26] = 255
        test_loader.data[i][25][25] = 255
        test_loader.data[i][24][26] = 255
        test_loader.data[i][26][24] = 255
        #test_loader.targets[i] = 9

    data_loader_test2 = torch.utils.data.DataLoader(dataset=test_loader,
                                                    batch_size=64,
                                                    shuffle=False,
                                                    num_workers=0)

    test(model, device, data_loader_test2)


    # 用于保存所有测试样本的真实标签和模型的预测结果
    true_labels = []
    predicted_labels = []

    # 查看并测试多个植入后门的测试样本
    for i, (sample, label) in enumerate(data_loader_test2):
        plt.imshow(sample[0][0].cpu().numpy(), cmap='gray')
        plt.show()

        # 将样本传递到模型中进行预测
        sample = sample.to(device)
        output = model(sample)
        pred = output.argmax(dim=1)

        # 保存每个样本的真实标签和预测结果
        true_labels.extend(label.cpu().numpy())  # 将真实标签添加到列表中
        predicted_labels.extend(pred.cpu().numpy())  # 将预测结果添加到列表中

        # 输出模型对该样本的预测结果
        print(f"Test Sample {i + 1} Original Label: ", label[0].item())
        print(f"Test Sample {i + 1} Predicted Label: ", pred[0].item())

        # 如果你只想查看前几个批次的结果，可以加一个限制条件
        if i >= 9:  # 例如只查看前10个批次
            break

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # 可视化混淆矩阵，并将标签修改为 0-9
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=True,
                xticklabels=np.arange(10),  # 横轴标签 0-9
                yticklabels=np.arange(10))  # 纵轴标签 0-9
    plt.title('Confusion Matrix of Backdoor Attack')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


if __name__=='__main__':
    main()
