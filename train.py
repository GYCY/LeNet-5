import torch
from torch import nn
from net import NET
from torch.optim import lr_scheduler # 从优化器模块中导入学习率调度器模块
from torchvision import datasets, transforms
import os

#数据转化为tensor格式
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])

#加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

#加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

#GPU计算
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net = NET()
net.to(device)

#定义损失函数（交叉熵）和优化器（SGD）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#定义学习率衰减策略 10 -> 0.1
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#训练函数
def train(dataloader, net, criterion, optimizer):
    net.train() # 设置为训练模式
    loss, correct , n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        #前向传播
        X, y = X.to(device), y.to(device)
        output = net(X)
        current_loss = criterion(output, y)
        _, pred = torch.max(output, axis=1)
        current_correct = torch.sum(pred == y)/output.size(0)

        #反向传播
        optimizer.zero_grad()
        current_loss.backward() # 计算梯度
        optimizer.step() # 根据梯度更新参数

        loss += current_loss.item()
        correct += current_correct.item()
        n += 1

    print(f"Train: Loss: {loss / n:.4f}, Accuracy: {correct / n*100:.4f}%")

#验证函数
def val(dataloader, net, criterion):
    net.eval() # 设置为评估模式
    loss, correct , n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = net(X)
            current_loss = criterion(output, y)
            _, pred = torch.max(output, axis=1)
            current_correct = torch.sum(pred == y)/output.size(0)

            loss += current_loss.item()
            correct += current_correct.item()
            n += 1

    print(f"Val: Loss: {loss / n:.4f}, Accuracy: {correct / n*100:.4f}%")
    return correct / n

#训练和验证
epoch = 50
max_acc = 0.0
for t in range(epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, net, criterion, optimizer)
    a = val(test_loader, net, criterion)
    #保存最优模型
    if a > max_acc:
        folder = "best_model"
        if not os.path.exists(folder):
            os.makedirs(folder)
        max_acc = a
        print(f"Saving model with accuracy: {max_acc*100:.4f}%")
        torch.save(net.state_dict(), f"{folder}/best_model.pth")
    scheduler.step() # 更新学习率

print("Done!")