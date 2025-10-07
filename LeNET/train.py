import torch
from torch import nn
import os
from torch import optim
from model import LeNET
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    data_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data',train=True,transform=data_transforms,download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_dataset = datasets.MNIST(root='./data',train=False,transform=data_transforms,download=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNET().to(device)
    SGD = optim.SGD(model.parameters(), lr=0.01)
    Loss_fn = nn.CrossEntropyLoss()
    all_epoch = 10
    total_train_step = 0
    for epoch in range(all_epoch):
        print(f'第{epoch+1}轮训练开始')
        for x, label in train_dataloader:
            x, label = x.to(device), label.to(device)
            predict_number = model(x)
            loss = Loss_fn(predict_number,label)
            SGD.zero_grad()
            loss.backward()
            SGD.step()
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print(f'训练次数为{total_train_step},损失为{loss}')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label in test_dataloader:
            x, label = x.to(device), label.to(device)
            outputs = model(x)
            loss = Loss_fn(outputs, label)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')

    folder = 'save_model'
    # path.exists：判断括号里的文件是否存在，存在为True，括号内可以是文件路径
    if not os.path.exists(folder):
        # os.mkdir() ：用于以数字权限模式创建目录
        os.mkdir('save_model')
    print('save best model')
    # torch.save(state, dir)保存模型等相关参数，dir表示保存文件的路径+保存文件名
    # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
    torch.save(model.state_dict(), 'save_model/best_model.pth')






