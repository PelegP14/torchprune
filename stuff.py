import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import src.torchprune.torchprune as tp
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res = np.load("C:\\Users\\Hanich\\Documents\\cluster_files\\resnet20_CIFAR10_e182_re0_retrain_int1_CIFAR10.npz")

class BaseLinear(torch.nn.Module):
    def __init__(self,indim = 48,outdim = 12):
        super().__init__()
        self.layer = torch.nn.Linear(indim,outdim)

    def forward(self,x):
        return self.layer(x)

class BaseConv(torch.nn.Module):
    def __init__(self,indim=1,outdim=12,pic_size=36):
        super().__init__()
        self.pic_size=pic_size
        self.conv1 = torch.nn.Conv2d(indim,indim*3,(3,3),padding=1)
        self.activation = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(3*indim,indim,(3,3),padding=1)
        self.layer = torch.nn.Linear(pic_size,outdim)

    def forward(self,x):
        x = self.activation(self.conv1(x.reshape(-1,1,int(self.pic_size**0.5),int(self.pic_size**0.5))))
        x = self.activation(self.conv2(x))
        x = self.layer(x.reshape(-1,self.pic_size))
        return x



data,label = make_blobs(500,n_features=36,centers=12)
data = data.astype(np.float32)
label = label.astype(np.long)
data,test_data,label,test_label = train_test_split(data,label,test_size=0.33)
data = torch.tensor(data)
label = torch.tensor(label)
train_set = TensorDataset(data,label)
train_loader = DataLoader(train_set,batch_size=32,shuffle=True)
test_data = torch.tensor(test_data)
test_label = torch.tensor(test_label)
test_set = TensorDataset(test_data, test_label)
test_loader = DataLoader(test_set, batch_size=32)
# transform_train = [
#     torchvision.transforms.Pad(4),
#     torchvision.transforms.RandomCrop(32),
#     torchvision.transforms.RandomHorizontalFlip(),
# ]
# transform_static = [
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
#     ),
# ]
#
#
# testset = torchvision.datasets.CIFAR10(
#     root="./local",
#     train=False,
#     download=True,
#     transform=tp.util.transforms.SmartCompose(transform_static),
# )
#
# trainset = torchvision.datasets.CIFAR10(
#     root="./local",
#     train=True,
#     download=True,
#     transform=tp.util.transforms.SmartCompose(
#         transform_train + transform_static
#     ),
# )
#
# size_s = 128
# batch_size = 128
# testset, set_s = torch.utils.data.random_split(
#     testset, [len(testset) - size_s, size_s]
# )
#
# loader_s = torch.utils.data.DataLoader(set_s, batch_size=32, shuffle=False)
# test_loader = torch.utils.data.DataLoader(
#     testset, batch_size=batch_size, shuffle=False
# )
# train_loader = torch.utils.data.DataLoader(
#     trainset, batch_size=batch_size, shuffle=False
# )

net = BaseConv() #tp.util.models.resnet20().to(device)
net = tp.util.net.NetHandle(net, "base")
optim = torch.optim.Adam(net.parameters(),lr=0.001)#torch.optim.SGD(net.parameters(),lr=0.1,weight_decay=1.0e-4,nesterov=True,momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,milestones=[150, 225],gamma=0.1)
for epoch in range(20):
    for data,label in train_loader:
        out = net(data.to(device))
        loss = torch.nn.functional.cross_entropy(out,label.to(device).long())
        optim.zero_grad()
        loss.backward()
        optim.step()
    count = 0
    success = 0
    for data,label in test_loader:
        out = net(data.to(device)).cpu()
        pred = torch.argmax(out,dim=1)
        success += torch.count_nonzero(pred==label)
        count += label.shape[0]
    scheduler.step()
    print("epoch {}: {:.3f}".format(epoch,success/count))

kr = 0.5
nets = [tp.MessiNetEfficient]
vals = []
for i,option in enumerate(nets):
    print("\n\n\n\n\n\n\n\n")
    comp_net = option(net,train_loader,None)
    comp_net.compress(keep_ratio=kr)
    print("size before: {}".format(comp_net.size()))
    val = [0,0,0,0]
    val[1] = comp_net.name
    val[2] = comp_net.size()
    val[3] = comp_net.flops()
    count = 0
    success = 0
    for data, label in test_loader:
        out = comp_net(data.to(device)).cpu()
        pred = torch.argmax(out, dim=1)
        success += torch.count_nonzero(pred == label)
        count += label.shape[0]
    val[0] = success / count
    vals.append(val)
    comp_net2 = option(net,train_loader,None)
    comp_net2.load_state_dict(comp_net.state_dict(), strict=False)
    comp_net2.load_state_dict(comp_net.state_dict(), strict=False)
    count = 0
    success = 0
    for data, label in test_loader:
        out = comp_net2(data.to(device)).cpu()
        pred = torch.argmax(out, dim=1)
        success += torch.count_nonzero(pred == label)
        count += label.shape[0]
    print(f"loaded net success {success/count} and size {comp_net2.size()}")
vals.sort(reverse=True,key=lambda elem:elem[0])
for i,item in enumerate(vals):
    print("{} got accuracy {} which landed it in {} place".format(item[1],item[0],i))
    print("it had {} params and {} flops".format(item[2],item[3]))
# temp_net = tp.TempNetALDSerrorComparison(net, train_loader, None)
# temp_net.compress(keep_ratio=0.5)
# # my_messi_net = tp.TempNet(net,train_loader,None)
# # my_messi_net.compress(keep_ratio=0.5)
# print("\n\n\n\n\n")
# alds_net = tp.TempNetALDSerrorSmartJMean(net, train_loader, None)
# alds_net.compress(keep_ratio=0.5)
# print("temp params:{}\nalds params:{}".format(temp_net.compressed_net.size(), alds_net.compressed_net.size()))
# print("temp flops:{}\nalds flops:{}".format(temp_net.compressed_net.flops(), alds_net.compressed_net.flops()))
# count = 0
# success = 0
# for data,label in test_loader:
#     out = alds_net(data.to(device)).cpu()
#     pred = torch.argmax(out,dim=1)
#     success += torch.count_nonzero(pred==label)
#     count += label.shape[0]
# print("alds: {:.3f}".format(success/count))
# count = 0
# success = 0
# for data,label in test_loader:
#     out = temp_net(data.to(device)).cpu()
#     pred = torch.argmax(out,dim=1)
#     success += torch.count_nonzero(pred==label)
#     count += label.shape[0]
# print("temp: {:.3f}".format(success/count))

