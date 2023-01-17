import torch
import numpy as np
import torchprune as tp
import torchvision
# tp.util.train.load_checkpoint()
net_handle = tp.util.net.NetHandle(tp.util.models.resnet20(num_classes=10))
base_net = tp.ReferenceNet(net_handle)
base_net_dict = torch.load("C:\\Users\\Hanich\\Documents\\cluster_files\\retrained_networks_avg_exp\\resnet20_CIFAR10_e182_ReferenceNet_0_p51_rep0_re0_i0"
                       ,map_location=torch.device('cpu'))['net']
base_net.load_state_dict(base_net_dict, strict=False)
jopt = tp.TempNetJOpt(net_handle,None,None)
jopt_dict = torch.load("C:\\Users\\Hanich\\Documents\\cluster_files\\retrained_networks_avg_exp\\resnet20_CIFAR10_e182_TempNetJOpt_0_p51_rep0_re0_i0"
                       ,map_location=torch.device('cpu'))['net']
jopt.load_state_dict(jopt_dict, strict=False)
alds_j = tp.TempNetJOpt(net_handle,None,None)
alds_j_dict = torch.load("C:\\Users\\Hanich\\Documents\\cluster_files\\retrained_networks_avg_exp\\resnet20_CIFAR10_e182_TempNetALDSerrorJOpt_0_p51_rep0_re0_i0"
                       ,map_location=torch.device('cpu'))['net']
alds_j.load_state_dict(alds_j_dict, strict=False)
example = jopt.compressed_net.torchnet.layer1[0].conv1
jopt_error = []
alds_j_error = []
ord = 2
# successes = [0,0,0]
# counts = [0,0,0]
# transform_static = [
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(
#         (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
#     ),
# ]
# batch_size = 128
#
# testset = torchvision.datasets.CIFAR10(
#     root="./local",
#     train=False,
#     download=True,
#     transform=tp.util.transforms.SmartCompose(transform_static),
# )
# test_loader = torch.utils.data.DataLoader(
#     testset, batch_size=batch_size, shuffle=False
# )
#
# for data, label in test_loader:
#     for i,net in enumerate([base_net,jopt,alds_j]):
#         out = net(data).cpu()
#         pred = torch.argmax(out, dim=1)
#         successes[i] += torch.count_nonzero(pred == label)
#         counts[i] += label.shape[0]
# print([success/count for success,count in zip(successes,counts)])
count_jopt = 0
count_alds_j = 0
for name, original_layer in base_net.compressed_net.torchnet.named_modules():
    if original_layer in base_net.compressed_net.compressible_layers:
        alds_j_layer = tp.method.base_decompose.base_decompose_util.get_attr(alds_j.compressed_net.torchnet,name.split('.'))
        scheme = tp.method.base_decompose.base_decompose_util.FoldScheme(0)
        if isinstance(alds_j_layer, tp.method.base_decompose.base_decompose_util.ProjectedModule):
            alds_j_layer = alds_j_layer.get_original_module()
        alds_j_weight = scheme.fold(alds_j_layer.weight)
        jopt_layer = tp.method.base_decompose.base_decompose_util.get_attr(jopt.compressed_net.torchnet,name.split('.'))
        if isinstance(jopt_layer, tp.method.base_decompose.base_decompose_util.ProjectedModule):
            jopt_layer = jopt_layer.get_original_module()
        jopt_weight = scheme.fold(jopt_layer.weight)

        original_weight = scheme.fold(original_layer.weight)
        jopt_error.append((torch.linalg.norm(jopt_weight-original_weight,ord=ord)/torch.linalg.norm(original_weight,ord=ord)).item())
        alds_j_error.append(
                (torch.linalg.norm(alds_j_weight - original_weight, ord=ord) / torch.linalg.norm(original_weight, ord=ord)).item())

print(count_jopt,count_alds_j)
result = np.vstack((jopt_error,alds_j_error))
print(result)
# print(base_net.torchnet)
# print(alds_j.compressed_net.torchnet)
# print(jopt.compressed_net.torchnet)