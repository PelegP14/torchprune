from scipy.spatial.transform import Rotation as R
import numpy as np
from tqdm import tqdm

def build_mat(mat_list, shape, partition):
    mat = np.zeros(shape)
    for i,m in enumerate(mat_list):
        mat[partition == i] = m
    return mat

def calculate_j_s(A_list,k,j,can_improve):
    singular_array = get_singular_values(A_list)
    j_list = [j]*k
    idx_to_increase, idx_to_decrease, improvement_is_possible = can_improve(singular_array,j_list)
    while idx_to_increase != idx_to_decrease and improvement_is_possible:
        print("improving j")
        j_list[idx_to_increase] += 1
        j_list[idx_to_decrease] -= 1
        idx_to_increase, idx_to_decrease, improvement_is_possible = can_improve(singular_array, j_list)
    return j_list

def get_singular_values(A_list):
    k = len(A_list)
    max_j  = min(A_list[0].shape[0], A_list[0].shape[1])
    singular_array = np.zeros((k,max_j+1))
    for i,A_i in enumerate(A_list):
        singular_array[i,:max_j] = np.linalg.svd(A_i, compute_uv=False)
    return singular_array

def can_improve_max(singular_array,j_list):
    current_sv = np.array([singular_array[i,j_list[i]] for i in range(len(j_list))])
    lower_sv = np.array([singular_array[i,j_list[i]-1] if j_list[i] > 0 else np.inf for i in range(len(j_list))])
    idx_to_increase = np.argmax(current_sv)
    idx_to_decrease = np.argmin(lower_sv)
    return idx_to_increase, idx_to_decrease, current_sv[idx_to_increase] > lower_sv[idx_to_decrease]

def random_subset(A,k):
    subset = np.random.choice(np.arange(A.shape[1]),size=A.shape[1]//k,replace=False)
    A_tag = A[:,subset]
    return A_tag

def get_stats_for_random_selections(A,k,samples,j):
    sum1 = 0
    sum2 = 0
    min_s = np.inf
    for i in tqdm(range(samples)):
        A_tag = random_subset(A,k)
        sigma = np.linalg.svd(A_tag, compute_uv=False)[j]
        sum1 += sigma
        sum2 += sigma**2
        min_s = np.minimum(min_s,sigma)
    avg = sum1/samples
    std = np.sqrt(sum2/samples-avg**2)
    print(f"stats:\navg:{avg}\nstd:{std}\nmin:{min_s}")


k = 2
num_sing_values = 3
A_list = []
partition = np.array([1,1,0,0,1,0])
shape = (partition.shape[0],num_sing_values)
for i in range(k):
    mats = R.random(2).as_matrix()
    s = np.diag([(i+2)**(j) for j in range(num_sing_values,0,-1)])
    A_list.append(mats[0]@s@mats[1])

A = build_mat(A_list,shape,partition)
j=1
B_list = []
for A_i in A_list:
    u, s, vt = np.linalg.svd(A_i)
    s[j:] = 0
    B_list.append(u @ np.diag(s) @ vt)

B = build_mat(B_list,shape,partition)

print("alds error {}".format(np.linalg.norm(A-B,ord=2)/np.linalg.norm(A,ord=2)))

j_s = calculate_j_s(A_list,k,j,can_improve_max)
C_list = []
for i,A_i in enumerate(A_list):
    u, s, vt = np.linalg.svd(A_i)
    s[j_s[i]:] = 0
    C_list.append(u @ np.diag(s) @ vt)

C = build_mat(C_list,shape,partition)

print("my error {}".format(np.linalg.norm(A-C,ord=2)/np.linalg.norm(A,ord=2)))
