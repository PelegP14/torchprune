import torch_rbf as rbf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from time import time
# Defining an RBF network class
from scipy.stats import ortho_group

from CORESET import obtainSensitivity, generateCoreset
trials = 10

class MyDataset(Dataset):
    
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        w = self.w[idx]
        return (x, y, w)


class Network(nn.Module):
    
    def __init__(self, layer_widths, layer_centres, basis_func):
        super(Network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(rbf.RBF(layer_widths[i], layer_centres[i], basis_func))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i+1]))
    
    def forward(self, x, w = None):
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return torch.multiply(out.T, w).T if w is not None else out
    
    def fit(self, x, y, w, epochs, batch_size, lr, loss_func):
        self.train()
        obs = x.size(0)
        trainset = MyDataset(x, y, w)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        epoch = 0
        while epoch < epochs:
            epoch += 1
            current_loss = 0
            batches = 0
            progress = 0
            for x_batch, y_batch, w_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                y_hat = self.forward(x_batch, w_batch)
                loss = loss_func(y_hat, y_batch)
                current_loss += (1/batches) * (loss.item() - current_loss)
                loss.backward()
                optimiser.step()
                progress += y_batch.size(0)
                sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f      ' % \
                                 (epoch, progress, obs, current_loss))
                sys.stdout.flush()



# Generating a dataset for a given decision boundary

x1 = np.linspace(-1, 1, 101)
x2 = 0.5*np.cos(np.pi*x1) + 0.5*np.cos(4*np.pi*(x1+1)) # <- decision boundary

samples = 20000
x = np.random.uniform(-1, 1, (samples, 2))
for i in range(samples):
    if i < samples//2:
        x[i,1] = np.random.uniform(-1, 0.5*np.cos(np.pi*x[i,0]) + 0.5*np.cos(4*np.pi*(x[i,0]+1)))
    else:
        x[i,1] = np.random.uniform(0.5*np.cos(np.pi*x[i,0]) + 0.5*np.cos(4*np.pi*(x[i,0]+1)), 1)

w = np.ones((x.shape[0], ))

x = np.random.randn(samples, 2) #100

approxMVEE = True

time_sens = time()
if approxMVEE:
    sensitivity = obtainSensitivity(np.hstack((x, np.linalg.norm(x, axis=1, ord=2)[:, np.newaxis],)),
                                    np.ones((x.shape[0], )), approxMVEE=approxMVEE)
else:
    sensitivity =\
        obtainSensitivity(np.hstack((x, np.linalg.norm(x, axis=1, ord=2)[:, np.newaxis], np.ones((x.shape[0], 1)))),
                          np.ones((x.shape[0], )), approxMVEE=True)
time_sens_end = time()

print('Sensitivity computation took {} seconds'.format(time_sens_end - time_sens))

steps = 100
x_span = np.linspace(-1, 1, steps)
y_span = np.linspace(-1, 1, steps)
xx, yy = np.meshgrid(x_span, y_span)
values = np.hstack((xx.ravel().reshape(xx.ravel().shape[0], 1),
                   yy.ravel().reshape(yy.ravel().shape[0], 1)))


#cost = lambda Q: np.sin(Q.dot(np.array([1, 3]))) + 2
#cost = (lambda Q: np.multiply(np.sin(Q.dot(np.array([np.pi, -2.21565]))), 1/Q.dot(np.array([np.pi, -2.21565])) ))
z = np.random.randn(2,1).flatten()
 
cost = lambda Q: np.abs(Q.dot(z / np.linalg.norm(z)))
print('cost = lambda Q: np.abs(Q.dot(z / np.linalg.norm(z)))')
#cost = lambda Q: Q[:,0] ** 2 / 4 + Q[:,1] ** 2 / 16
#print('cost = lambda Q: Q[:,0] ** 2 / 4 + Q[:,1] ** 2 / 16')
tx = torch.from_numpy(x).float()
ty = torch.cat((torch.zeros(samples//2,1), torch.ones(samples//2,1)), dim=0)
tw = torch.from_numpy(np.ones((x.shape[0], ))).float()
ty = torch.from_numpy(cost(x)[:, np.newaxis]).float()





# Instanciating and training an RBF network with the Gaussian basis function
# This network receives a 2-dimensional input, transforms it into a 40-dimensional
# hidden representation with an RBF layer and then transforms that into a
# 1-dimensional output/prediction with a linear layer

# To add more layers, change the layer_widths and layer_centres lists

layer_widths = [2, 1]
layer_centres = [40]
basis_func = rbf.gaussian

target = cost(values)
loss_func = nn.MSELoss() # nn.BCEWithLogitsLoss()

start_whole = time()
rbfnet = Network(layer_widths, layer_centres, basis_func)
rbfnet.fit(tx, ty, tw, 1000, samples, 0.01, loss_func)
rbfnet.eval()
end_whole_training = time()

print('Took {} seconds to train RBFNN on whole data'.format(end_whole_training - start_whole))
# Plotting the ideal and learned decision boundaries

with torch.no_grad():
    preds_all = (torch.sigmoid(rbfnet(torch.from_numpy(values).float()))).cpu().detach().numpy()



vals = np.empty((2, trials))
times = np.empty((2, trials))

for trial in range(trials):
    x_core, y_core, w_core, time_taken_coreset = generateCoreset(X=x, y=ty.cpu().detach().numpy(),
                                                                 sensitivity=sensitivity,
                                                                 sample_size=x.shape[0] // 10, weights=w)

    x_uni, y_uni, w_uni, time_taken_uni = generateCoreset(X=x, y=ty.cpu().detach().numpy(),
                                                          sensitivity=np.ones(sensitivity.shape),
                                                          sample_size=x.shape[0] // 10, weights=w)

    # Plotting the ideal and learned decision boundaries
    start_coreset = time()
    rbfnet_core = Network(layer_widths, layer_centres, basis_func)
    rbfnet_core.fit(torch.from_numpy(x_core).float(), torch.from_numpy(y_core).float(),
                    torch.from_numpy(np.ones(w_core.shape)).float(), 1000, x_core.shape[0]//10, 0.01, loss_func)
    rbfnet_core.eval()
    end_whole_coreset = time()

    with torch.no_grad():
        preds_coreset = (torch.sigmoid(rbfnet_core(torch.from_numpy(values).float()))).cpu().detach().numpy()



    print('Took {} seconds to train RBFNN on our coreset'.format(end_whole_coreset - start_coreset))


    start_uniform = time()
    rbfnet_uniform = Network(layer_widths, layer_centres, basis_func)
    rbfnet_uniform.fit(torch.from_numpy(x_uni).float(), torch.from_numpy(y_uni).float(),
                    torch.from_numpy(np.ones(w_uni.shape)).float(), 1000, x_uni.shape[0] //10, 0.01, loss_func)
    rbfnet_uniform.eval()
    end_whole_uniform = time()

    with torch.no_grad():
        preds_uniform = (torch.sigmoid(rbfnet_uniform(torch.from_numpy(values).float()))).cpu().detach().numpy()
    
    print('Took {} seconds to train RBFNN on uniform coreset'.format(end_whole_uniform - start_uniform))
    

    vals[0, trial] = 1/target.shape[0] * np.linalg.norm(target.flatten() - preds_uniform.flatten(), ord=2) ** 2
    vals[1, trial] = 1/target.shape[0] * np.linalg.norm(target.flatten() - preds_coreset.flatten(), ord=2) ** 2
    times[0, trial] = end_whole_uniform - start_uniform
    times[0, trial] = end_whole_coreset - start_coreset

print(vals)
val_means = np.mean(vals, axis=1)
time_means = np.mean(times, axis=1)
print('\n MEAN RMSE between us and target is {}'.format(val_means[1]))
print('MEAN RMSE between uniform and target is {}'.format(val_means[0]))
print('RMSE between all and target is {}'.format(1/target.shape[0] * np.linalg.norm(target.flatten() - preds_all.flatten(), ord=2) ** 2))
# print('RMSE between all data and target{}'.format(np.linalg.norm(target.flatten() - preds_all.flatten(), ord=2) ** 2))
# print('RMSE between us and target {}'.format(1/target.shape[0] * np.linalg.norm(target.flatten() - preds_coreset.flatten(), ord=2) ** 2))
# print('RMSE between uniform and target {}'.format(1/target.shape[0] * np.linalg.norm(target.flatten() - preds_uniform.flatten(), ord=2) ** 2))




# ideal_0 = values[np.where(values[:,1] <= 0.5*np.cos(np.pi*values[:,0]) + 0.5*np.cos(4*np.pi*(values[:,0]+1)))[0]]
# ideal_1 = values[np.where(values[:,1] > 0.5*np.cos(np.pi*values[:,0]) + 0.5*np.cos(4*np.pi*(values[:,0]+1)))[0]]
# area_0 = values[np.where(preds[:, 0] <= 0.5)[0]]
# area_1 = values[np.where(preds[:, 0] > 0.5)[0]]
#
# fig, ax = plt.subplots(figsize=(16,8), nrows=1, ncols=2)
# ax[0].scatter(x[:samples//2,0], x[:samples//2,1], c='dodgerblue')
# ax[0].scatter(x[samples//2:,0], x[samples//2:,1], c='orange', marker='x')
# ax[0].scatter(ideal_0[:, 0], ideal_0[:, 1], alpha=0.1, c='dodgerblue')
# ax[0].scatter(ideal_1[:, 0], ideal_1[:, 1], alpha=0.1, c='orange')
# ax[0].set_xlim([-1,1])
# ax[0].set_ylim([-1,1])
# ax[0].set_title('Ideal Decision Boundary')
# ax[1].scatter(x[:samples//2,0], x[:samples//2,1], c='dodgerblue')
# ax[1].scatter(x[samples//2:,0], x[samples//2:,1], c='orange', marker='x')
# ax[1].scatter(area_0[:, 0], area_0[:, 1], alpha=0.1, c='dodgerblue')
# ax[1].scatter(area_1[:, 0], area_1[:, 1], alpha=0.1, c='orange')
# ax[1].set_xlim([-1,1])
# ax[1].set_ylim([-1,1])
# ax[1].set_title('RBF Decision Boundary')
# plt.show()