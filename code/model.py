from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 加载mnist数据集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=10, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=10, shuffle=True)

# 超参数设置
batch_size = 10
epoch = 1
learning_rate = 0.001
# 生成对抗样本的个数
adver_nums = 1000


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 选择设备
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# 初始化网络，并定义优化器
simple_model = Net().to(device)
optimizer1 = torch.optim.SGD(simple_model.parameters(),lr = learning_rate,momentum=0.9)
print (simple_model)


# 训练模型
def train(model,optimizer):
  for i in range(epoch):
    for j,(data,target) in tqdm(enumerate(train_loader)):
      data = data.to(device)
      target = target.to(device)
      logit = model(data)
      loss = F.nll_loss(logit,target)
      model.zero_grad()
      # 如下：因为其中的loss是单个tensor就不能用加上一个tensor的维度限制
      loss.backward()
      optimizer.step()
      if j % 1000 == 0:
        print ('第{}个数据，loss值等于{}'.format(j,loss))
train(simple_model,optimizer1)

# eval
# 训练完模型后，加上固定DROPOUT层
simple_model.eval()

# 模型测试
def test(model,name):
  correct_num = torch.tensor(0).to(device)
  for j,(data,target) in tqdm(enumerate(test_loader)):
    data = data.to(device)
    target = target.to(device)
    logit = model(data)
    pred = logit.max(1)[1]
    num = torch.sum(pred==target)
    correct_num = correct_num + num
  print (correct_num)
  print ('\n{} correct rate is {}'.format(name,correct_num/10000))
test(simple_model,'simple model')


from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

def PGD(model):
  adver_example = None
  adver_target = None
  clean_example = None
  clean_target = None
  for i,(data,target) in enumerate(test_loader):
    if i>=1:
      break

    adver_example = projected_gradient_descent(model, data.to(device),0.15,0.02,10,np.inf)
    adver_target = torch.max(model(adver_example),1)[1]
    clean_example = data
    clean_target = target
  return adver_example,adver_target,clean_example,clean_target,'PGD attack'

def FGSM(model):
  adver_example = None
  adver_target = None
  clean_example = None
  clean_target = None
  for i,(data,target) in enumerate(test_loader):
    if i>=1:
      break

    adver_example = fast_gradient_method(model, data.to(device), 0.1, np.inf)
    adver_target = torch.max(model(adver_example),1)[1]
    clean_example = data
    clean_target = target
  return adver_example,adver_target,clean_example,clean_target,'FGSM attack'

def CW(model):
  adver_example = None
  adver_target = None
  clean_example = None
  clean_target = None
  for i,(data,target) in enumerate(test_loader):
    if i>=1:
      break

    adver_example = carlini_wagner_l2(model, data.to(device), 10, y = torch.tensor([3]*batch_size, device = device) ,targeted = True)
    adver_target = torch.max(model(adver_example),1)[1]
    clean_example = data
    clean_target = target
  return adver_example,adver_target,clean_example,clean_target,'CW attack'

def plot_clean_and_adver(adver_example,adver_target,clean_example,clean_target,attack_name):
  n_cols = 2
  n_rows = 5
  cnt = 1
  cnt1 = 1
  plt.figure(figsize=(4*n_rows,2*n_cols))
  for i in range(n_cols):
    for j in range(n_rows):
      plt.subplot(n_cols,n_rows*2,cnt1)
      plt.xticks([])
      plt.yticks([])
      if j == 0:
        plt.ylabel(attack_name,size=15)
      plt.title("{} -> {}".format(clean_target[cnt-1], adver_target[cnt-1]))
      plt.imshow(clean_example[cnt-1].reshape(28,28).to('cpu').detach().numpy(),cmap='gray')
      plt.subplot(n_cols,n_rows*2,cnt1+1)
      plt.xticks([])
      plt.yticks([])
      plt.imshow(adver_example[cnt-1].reshape(28,28).to('cpu').detach().numpy(),cmap='gray')
      cnt = cnt + 1
      cnt1 = cnt1 + 2
  plt.show()
  print ('\n')

adver_example,adver_target,clean_example,clean_target,attack_name= FGSM(simple_model)
plot_clean_and_adver(adver_example,adver_target,clean_example,clean_target,attack_name)


adver_example,adver_target,clean_example,clean_target,attack_name= CW(simple_model)
plot_clean_and_adver(adver_example,adver_target,clean_example,clean_target,attack_name)


adver_example,adver_target,clean_example,clean_target,attack_name= PGD(simple_model)
plot_clean_and_adver(adver_example,adver_target,clean_example,clean_target,attack_name)
