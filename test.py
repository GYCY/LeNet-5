import torch
from torch import nn
from net import NET
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

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

net.load_state_dict(torch.load('./best_model/best_model.pth')) # 加载模型参数

#结果
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

#tensor转为PIL图像
show = ToPILImage()

#进入验证
net.eval()
for i in range(20):
    X, y = test_dataset[i][0], test_dataset[i][1]
    show(X).show() # 显示图像

    X = Variable(X.view(1, 1, 28, 28)).to(device)
    with torch.no_grad():
        pred = net(X)
        predicted, actual = classes[torch.argmax(pred)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')