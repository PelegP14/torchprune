import numpy as np
from scipy.optimize import linprog
import cvxpy as cp

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(problem=None,costs=None,errors_vec=None,max_error_param=None,vec_sum=None,selection=None,n=-1,m=-1)
def low_cost_average(errors,cost_per_row,max_error):
    pad = np.zeros(errors.shape[0]).reshape(-1,1)
    errors = np.hstack((errors, pad))
    m, n = errors.shape
    # create and compile a new problem if needed and store its params in static variables
    if low_cost_average.problem is None or low_cost_average.m != m or low_cost_average.n != n:
        low_cost_average.n = n
        low_cost_average.m = m
        low_cost_average.selection = cp.Variable(n*m,boolean=True)
        low_cost_average.costs = cp.Parameter(n*m)
        low_cost_average.errors_vec = cp.Parameter(n*m)
        low_cost_average.max_error_param = cp.Parameter()
        low_cost_average.vec_sum = cp.Parameter((m,n*m))
        obj = cp.Minimize(low_cost_average.costs @ low_cost_average.selection)
        constraints = [low_cost_average.errors_vec @ low_cost_average.selection <= low_cost_average.max_error_param,
                       low_cost_average.vec_sum @ low_cost_average.selection == 1]
        low_cost_average.problem = cp.Problem(obj, constraints)

    total_costs = (np.arange(n)+1)*cost_per_row[0]

    vec_entry_sum_mat = np.zeros((m,n*m))
    vec_entry_sum_mat[0,:n] = 1

    for i in range(1,m):
        total_costs = np.hstack((total_costs,(np.arange(n)+1)*cost_per_row[i]))
        vec_entry_sum_mat[i,i*n:(i+1)*n] = 1

    low_cost_average.costs.value = total_costs
    low_cost_average.errors_vec.value = errors.flatten()
    low_cost_average.max_error_param.value = max_error*m
    low_cost_average.vec_sum.value = vec_entry_sum_mat

    low_cost_average.problem.solve(solver=cp.GLPK_MI)
    res = low_cost_average.selection.value.reshape(errors.shape)
    # print("mean error = {} while max error = {}".format(errors[res==1].mean(),max_error))
    # print(res)
    # print(np.argmax(res,axis=1)+1)

    return np.argmax(res,axis=1)

@static_vars(problem=None,costs=None,errors=None,jk_param=None,selection=None,max_error=None,n=-1,m=-1)
def j_opt(errors,j):
    pad = np.zeros(errors.shape[0]).reshape(-1,1)
    errors = np.hstack((errors, pad))
    m, n = errors.shape
    # create and compile a new problem if needed and store its params in static variables
    if j_opt.problem is None or j_opt.m != m or j_opt.n != n:
        j_opt.n = n
        j_opt.m = m
        j_opt.selection = cp.Variable((m,n),boolean=True)
        j_opt.max_error = cp.Variable()
        j_opt.costs = cp.Parameter((m,n))
        j_opt.errors = cp.Parameter((m,n))
        j_opt.jk_param = cp.Parameter()
        obj = cp.Minimize(j_opt.max_error)
        constraints = [cp.trace(j_opt.costs @ j_opt.selection.T) == j_opt.jk_param,
                       cp.sum(j_opt.selection,axis=1) == 1,
                       cp.diag(j_opt.errors @ j_opt.selection.T) <= j_opt.max_error]
        j_opt.problem = cp.Problem(obj, constraints)

    total_costs = np.repeat((np.arange(n) + 1).reshape(-1,1),m,axis=1).T

    j_opt.costs.value = total_costs
    j_opt.errors.value = errors
    j_opt.jk_param.value = j*m

    j_opt.problem.solve(solver=cp.GLPK_MI)
    res = j_opt.selection.value
    # print("mean error = {} while max error = {}".format(errors[res==1].mean(),max_error))
    # print(res)
    # print(np.argmax(res,axis=1)+1)

    return np.argmax(res,axis=1)

def j_opt_old(j_s, errors):
    singular_array = errors
    idx_to_increase, idx_to_decrease, improvement_is_possible = can_improve_max(singular_array, j_s.tolist())
    while idx_to_increase != idx_to_decrease and improvement_is_possible:
        # print("improving j")
        j_s[idx_to_increase] += 1
        j_s[idx_to_decrease] -= 1
        idx_to_increase, idx_to_decrease, improvement_is_possible = can_improve_max(singular_array, j_s.tolist())
    return j_s


def can_improve_max(singular_array, j_list):
    current_sv = np.array([singular_array[i, j_list[i]] if j_list[i]<singular_array.shape[1] else 0 for i in range(len(j_list))])
    lower_sv = np.array([singular_array[i, j_list[i] - 1] if j_list[i] > 0 else np.inf for i in range(len(j_list))])
    idx_to_increase = np.argmax(current_sv)
    idx_to_decrease = np.argmin(lower_sv)
    return idx_to_increase, idx_to_decrease, current_sv[idx_to_increase] > lower_sv[idx_to_decrease]

def low_cost_average_linear(errors,cost_per_row,max_error):
    pad = np.zeros(errors.shape[0])
    errors = np.hstack((errors,pad))
    m, n = errors.shape

    y = cp.Variable(n * m, boolean=True)
    obj_func = (np.arange(n) + 1) * cost_per_row[0]

    lhs_eq = np.zeros((m, n * m))
    lhs_eq[0, :n] = 1
    rhs_eq = np.ones(m)

    lhs_ineq = errors.reshape(1, -1)
    rhs_ineq = np.array(max_error) * m

    bounds = np.vstack((np.zeros(n * m), np.ones(n * m))).T

    for i in range(1, m):
        obj_func = np.hstack((obj_func, (np.arange(n) + 1) * cost_per_row[i]))
        lhs_eq[i, i * n:(i + 1) * n] = 1
    res = linprog(
        c=obj_func,
        A_ub=lhs_ineq,
        b_ub=rhs_ineq,
        A_eq=lhs_eq,
        b_eq=rhs_eq,
        bounds=bounds
    )
    res = res.x.reshape(10, 5)


def greedy_optimization(errors,cost_per_row,max_error):
    obj_func_decrease = std_times_average
    pad = np.zeros(errors.shape[0]).reshape(-1, 1)
    errors = np.hstack((errors, pad))
    m, n = errors.shape

    if max_error <= 0:
        return np.ones(m,dtype=int)*(n-1)

    selection = np.zeros(m,dtype=int)
    current_errors = errors[np.arange(m),selection]
    while np.mean(current_errors) > max_error:
        options = np.array([errors[i,selection[i]+1] if selection[i]+1 < n else np.nan for i in range(m)])
        decreases = obj_func_decrease(current_errors,options)
        decreases /= cost_per_row
        decreases = np.ma.array(decreases,mask=np.isnan(decreases))
        best_option = np.argmax(decreases)
        selection[best_option] += 1
        current_errors = errors[np.arange(m), selection]

    return selection

def alds_optimization_max(errors,cost_per_row,max_error):

    # rank_j's correspond to index of 1st relative error smaller than arg.
    # now replace rank_j's with a more refined computation.
    bigger = errors > max_error

    # get ranks now per layer
    ret = bigger.sum(axis=-1)

    return ret


def std_times_average(current_errors,options):
    decreases = np.empty_like(current_errors)
    base_value = current_errors.mean()*current_errors.std()
    for i,option in enumerate(options):
        new_errors = current_errors.copy()
        new_errors[i] = option
        new_value = new_errors.mean()*new_errors.std()
        decreases[i] = base_value - new_value
    return decreases

if __name__ == "__main__":
    errors = np.random.random((10, 5))
    errors = -np.sort(-errors,axis=1)
    cost_per_row = np.random.randint(low=1,high=100,size=10)
    opt_res = greedy_optimization(errors, cost_per_row, 0.2)
    old_res = low_cost_average(errors, cost_per_row, 0.2)
    pad = np.zeros(errors.shape[0]).reshape((-1,1))
    errors = np.hstack((errors,pad))
    print("new -- j_s {} total j_s {} max error {} average error {} std {} total params {}".format(
        opt_res,
        np.sum(opt_res),
        np.max(errors[np.arange(10),opt_res]),
        np.mean(errors[np.arange(10),opt_res]),
        np.std(errors[np.arange(10),opt_res]),
        opt_res@cost_per_row))
    print("old -- j_s {} total j_s {} max error {} average error {} std {} total params {}".format(
        old_res,
        np.sum(old_res),
        np.max(errors[np.arange(10),old_res]),
        np.mean(errors[np.arange(10),old_res]),
        np.std(errors[np.arange(10),old_res]),
        old_res@cost_per_row))

    #### CHECK FOR MAX OPTIM VS HEURISTIC
    # check = True
    # for i in range(1000):
    #     errors = np.random.random((10, 5))
    #     errors = -np.sort(-errors,axis=1)
    #     opt_res = j_opt(errors,4)
    #     old_res = j_opt_old(np.array([3]*10),errors)
    #     check = (np.array(opt_res) == np.array(old_res)).all()
    #     if not check:
    #         break
    # print(check)
    # pad = np.zeros(errors.shape[0]).reshape((-1,1))
    # errors = np.hstack((errors,pad))
    # print("j_s {} total j_s {} max error {} average error {}".format(opt_res,np.sum(opt_res),np.max(errors[np.arange(10),opt_res]),np.mean(errors[np.arange(10),opt_res])))
    # print("j_s {} total j_s {} max error {} average error {}".format(old_res,np.sum(old_res),np.max(errors[np.arange(10),old_res]),np.mean(errors[np.arange(10),old_res])))

    #### CHECK FOR MIN COST AVG
    # for i in range(10):
    #     errors = np.random.random((10,5))
    #     errors = -np.sort(-errors,axis=1)
    #     cost_per_row = np.random.randint(low=90,high=100,size=10)
    #     print(low_cost_average(errors,cost_per_row,0.5))
    # for i in range(10):
    #     errors = np.random.random((7,9))
    #     errors = -np.sort(-errors,axis=1)
    #     cost_per_row = np.random.randint(low=90,high=100,size=7)
    #     print(low_cost_average(errors,cost_per_row,0.5))

