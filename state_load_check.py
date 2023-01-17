import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import src.torchprune.torchprune as tp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
data = torch.tensor(data).to(device)
label = torch.tensor(label).long().to(device)
train_set = TensorDataset(data,label)
train_loader = DataLoader(train_set,batch_size=32,shuffle=True)
test_data = torch.tensor(test_data).to(device)
test_label = torch.tensor(test_label).long().to(device)
test_set = TensorDataset(test_data, test_label)
test_loader = DataLoader(test_set, batch_size=32)

net = BaseConv().to(device) #tp.util.models.resnet20().to(device)
net = tp.util.net.NetHandle(net, "base").to(device)
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
        out = net(data.to(device))
        pred = torch.argmax(out,dim=1)
        success += torch.count_nonzero(pred==label.to(device))
        count += label.shape[0]
    scheduler.step()
    print("epoch {}: {:.3f}".format(epoch,success/count))

kr = 0.5
nets = [tp.MessiNetEfficient,tp.SnipNet,tp.SiPPNet,tp.TempNetALDSerrorJOpt,tp.ALDSNet,tp.MessiNet]
vals = []
for i,option in enumerate(nets):
    print("\n\n\n\n\n\n\n\n")
    comp_net = option(net,train_loader,torch.nn.CrossEntropyLoss()).to(device)
    comp_net.compress(keep_ratio=kr)
    print("size before: {}".format(comp_net.size()))
    val = [0,0,0,0]
    comp_net.to("cpu")
    comp_net2 = option(net.to("cpu"),train_loader,None)
    print("started creation")
    for name, module in comp_net2.named_modules():
        if hasattr(module, "weight"):
            print("module name {} type {} and device {}".format(name, module.__class__,
                                                                module.weight.device))
    print("finished creation \n\n")
    comp_net2.load_state_dict(comp_net.state_dict(),strict=False)
    comp_net2.to(device)
    comp_net.to(device)
    for data, label in test_loader:
        out1 = comp_net(data.to(device))
        out2 = comp_net2(data.to(device))
    optim = torch.optim.Adam(comp_net2.parameters(), lr=0.001)
    for data,label in train_loader:
        out = comp_net2(data.to(device))
        loss = torch.nn.functional.cross_entropy(out,label.to(device).long())
        optim.zero_grad()
        loss.backward()
        optim.step()
    comp_net.to("cpu")
    comp_net2.to("cpu")
    comp_net2.load_state_dict(comp_net.state_dict(),strict=False)
    for data, label in test_loader:
        out1 = comp_net(data.to("cpu"))
        out2 = comp_net2(data.to("cpu"))
    print("size of loaded net: {}".format(comp_net2.size()))
    val[1] = comp_net.name
    val[2] = comp_net.size()
    val[3] = comp_net.flops()
    count = 0
    success = 0
    comp_net.to(device)
    for data, label in test_loader:
        out = comp_net(data.to(device))
        pred = torch.argmax(out, dim=1)
        success += torch.count_nonzero(pred == label.to(device))
        count += label.shape[0]
    val[0] = success / count
    vals.append(val)

vals.sort(reverse=True,key=lambda elem:elem[0])
for i,item in enumerate(vals):
    print("{} got accuracy {} which landed it in {} place".format(item[1],item[0],i))
    print("it had {} params and {} flops".format(item[2],item[3]))