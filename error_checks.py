import torchprune as tp
import torchprune.method.temp.temp_util.factor as factor
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torch
import numpy as np


def get_coreset(weight, scheme, max_n):
    new_weight = scheme.fold(weight.detach())
    kernel_size = 1
    for size in scheme.get_kernel(weight):
        kernel_size *= size
    n = new_weight.shape[1] // kernel_size
    new_n = n
    for i in range(1, n):
        if n % i == 0 and (n // i) * kernel_size <= max_n:
            new_n = n // i
            break
    if new_n == n:
        return weight
    n *= kernel_size
    new_n *= kernel_size
    indxs = np.random.choice(np.arange(n), size=new_n, replace=False, p=np.ones(n) / n)
    new_weight = new_weight[:, torch.tensor(indxs, dtype=torch.long)]
    return scheme.unfold(new_weight, scheme.get_kernel(weight))

def rel_error_pc(weight,k_split,scheme):
    # calculate k,j projective clustering for each of the possible j values and with the given k
    # fold into matrix operator
    device = weight.device
    op_norm = tp.method.alds.ALDSErrorIterativeAllocator._compute_norm_for_weight(weight, scheme, ord=2)

    weight = scheme.fold(weight.detach()).t()

    # compute rank (notice the transpose when unfolding)
    rank_k = min(weight.shape[1], weight.shape[0] // k_split)

    rel_error = []
    for j in tqdm(range(rank_k), desc=f"layer of size n={weight.shape[0]}, d={weight.shape[1]}, k={k_split}",
                  leave=False):
        partition, list_u, list_v = factor.raw_messi(
            weight,
            j=j,
            k=k_split,
            verbose=False
        )
        u_stitched, v_stitched = factor.stitch(partition, list_u, list_v)

        new_weight = u_stitched @ v_stitched

        diff = new_weight - weight

        rel_error.append(torch.linalg.norm(diff, ord=2).item())

    # # compute flats and errors
    # rel_error = []
    # for j in range(0,rank_k):
    #     flats = factor.getProjectiveClustering(weight, j, k_split,verbose=False)
    #     rel_error.append(factor.getCost(weight, flats))

    rel_error = torch.tensor(rel_error, device=device) / (op_norm * np.sqrt(k_split))
    return rel_error

def rel_error_alds(weight,k_split,scheme):
    device = weight.device
    op_norm = tp.method.alds.ALDSErrorIterativeAllocator._compute_norm_for_weight(weight, scheme, ord=2)

    weight = scheme.fold(weight.detach()).t()
    n = weight.shape[0]

    # compute rank (notice the transpose when unfolding)
    rank_k = min(weight.shape[1], weight.shape[0] // k_split)

    partition = [i // int(n / k_split) for i in range(n)]
    idx_list = [[row for row in range(n) if partition[row] == z] for z in range(k_split)]
    A_list = [weight[idx, :] for idx in idx_list]
    SVD_list = [torch.svd(M) for M in A_list]

    rel_error = []
    for j in tqdm(range(rank_k), desc=f"layer of size n={weight.shape[0]}, d={weight.shape[1]}, k={k_split}",
                  leave=False):
        list_u = [U[:, :j].matmul(torch.diag(D[:j])) for U, D, _ in SVD_list]
        list_v = [V.t()[:j] for _,_,V in SVD_list]
        u_stitched, v_stitched = factor.stitch(partition, list_u, list_v)

        new_weight = u_stitched @ v_stitched

        diff = new_weight - weight

        rel_error.append(torch.linalg.norm(diff, ord=2).item())

    # # compute flats and errors
    # rel_error = []
    # for j in range(0,rank_k):
    #     flats = factor.getProjectiveClustering(weight, j, k_split,verbose=False)
    #     rel_error.append(factor.getCost(weight, flats))

    rel_error = torch.tensor(rel_error, device=device) / (op_norm * np.sqrt(k_split))
    return rel_error

def main():
    weight = torch.load("example.pth",map_location=torch.device('cpu'))
    scheme = tp.method.base_decompose.FoldScheme(0)
    coreset = get_coreset(weight, scheme, 144)
    # rel_error = rel_error_pc(weight,4,scheme)
    rel_error_coreset = rel_error_pc(coreset,4,scheme)
    rel_alds = rel_error_alds(weight,4,scheme)
    # strech = np.linspace(1,len(rel_error)-1,num=len(rel_error_coreset)-1).astype(int)
    # combine = rel_alds.clone()
    # combine[strech] = rel_error_coreset[1:]
    plt.figure()
    # plt.plot(range(len(rel_error)),rel_error,label="orig")
    plt.plot(range(len(rel_alds)), rel_alds, label="alds for entire matrix")
    plt.plot(range(len(rel_error_coreset)), rel_error_coreset, label="projective clustering for coreset")
    # plt.plot(range(len(combine)),combine,label = 'combine')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
