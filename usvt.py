import numpy as np

def usvt(A, eta = 0.0001):
    """
    Compute the USVT matrix imputations

    Parameters:
    ----------
    A : a N x T matrix with missing values denoted as nan
    eta : a small positive (or 0) that affects the singular value threshold.
            by default 0 

    Returns:
    ----------
    A_hat : the USVT estimate of A
    """
    # later should generalize for all matrices (i.e. transpose if m > n)

    N, T = A.shape

    # data prep
    # of observed values, 
    a = np.nanmin(A)
    b = np.nanmax(A)

    A_scaled = ((A - ((a + b) / 2)) / ((b-a)/2))
    Y_scaled = np.where(np.isnan(A_scaled), 0, A_scaled)

    p_hat = 1 - (np.count_nonzero(np.isnan(A)) / np.size(A))

    # currently follow Chatterjee 2015 threshold value
    threshold = (2 + eta) * np.sqrt(T * p_hat)

    u, s, vt = np.linalg.svd(Y_scaled, full_matrices=False)

    s_mask = s >= threshold
    # j_threshold = 0
    # while s[j_threshold] >= threshold:
    #     j_threshold += 1
    #     if j_threshold == len(s):
    #         break
    # W = 1/p_hat *  np.matrix(u[:, :j_threshold]) * np.diag(s[:j_threshold]) * np.matrix(vt[:j_threshold, :])
    # if np.sum(s_mask) == 1:
    #     W = np.outer(u[:, s_mask], vt[s_mask, :]) * s[s_mask] / p_hat
    # else:
    W = (u[:, s_mask] @ np.diag(s[s_mask]) @ vt[s_mask, :]) / p_hat

    # cap to [-1, 1]
    W = W.clip(-1, 1)
    # W[W>1]  = 1.0
    # W[W<-1] = -1.0

    # rescale:
    res = ((W*((b-a)/2)) + (a+b)/2)

    return res


# external implementation for verification
# def standardize(data, a=0, b=1):
#     shift = (a+b)/2
#     scale = (b-a)/2
#     return (data - shift) / scale 

# def undo(data, a=0, b=1):
#     shift = (a+b)/2
#     scale = (b-a)/2
#     return data * scale + shift

# def prop_obs(arr):
#     num_missing = np.count_nonzero(np.isnan(arr))
#     return 1 - num_missing/arr.size 

# def usvt(data_obs, save_path=None, action_list=None, verbose=False):
#     '''
#     Takes in normalized data data_obs, applies USVT to fill in the tensor
#     Args:
#         data_obs: N x T x A matrix, usvt is applied to every N x T slice 
#     '''
#     N, T = data_obs.shape[0], data_obs.shape[1]
#     # if action_list is None:
#     #     action_list = list(range(A))

#     # p_obs = []
#     # for aa, a in enumerate(action_list):
#     p_hat = prop_obs(data_obs)
#     # p_obs.append(p)
    
#     data_obs = np.ma.masked_array(data_obs, np.isnan(data_obs)) #mask_arr(data_obs)
#     # print("min, max", np.nanmax(data_obs), np.nanmin(data_obs))


#     # step 8: standardize data to [-1,1]
#     st_min, st_max = np.nanmin(data_obs), np.nanmax(data_obs)
#     arr = standardize(data_obs, st_min, st_max)

#     # step 1: fill in with 0s
#     arr = arr.filled(0)

#     # step 2: compute svd
#     u,s,vh = np.linalg.svd(arr)
    
#     # step 3: proportion of observed (precomputed)
#     # p_hat = p_obs[aa]
#     # print("p_hat", p_hat)
    
#     # step 4: choose threshold, compute set of thresholded values
#     q_hat = p_hat
#     eta = 0.0001
#     # var = 1/12 * (2*0.9)**2 + 0.3*0.3*2 + 0.05*0.05 + 0.001*0.001
#     # if var < 1:
#     #     print("variance", var)
#     #     q_hat = p_hat * var + p_hat * (1-p_hat) * (1-var)
#     #     print("q_hat", q_hat)
#     threshold = (2 + eta) * np.sqrt(T * q_hat)
#     if verbose:
#         print(s[:5])
#     j_threshold = 0
#     while s[j_threshold] >= threshold:
#         j_threshold += 1
#         if j_threshold == len(s):
#             break
#     if verbose:
#         print(f"Threshold {threshold}, at {j_threshold} SV")

#     # step 5: reconstruct
#     # W = 1/q_hat * 
#     W = 1/q_hat *  np.matrix(u[:, :j_threshold]) * np.diag(s[:j_threshold]) * np.matrix(vh[:j_threshold, :])
#     W = W.clip(-1,1)
#     W = undo(W, st_min, st_max)
#     # W = normalize(W, "max", data_obs_max)
#     # print("W shape", W.shape)
#     # est_list.append(np.asarray(W))
#     # if verbose:
#     #     print(len(est_list))
#     # est_stack = np.stack(est_list, axis=-1)
#     # if save_path:
#     #     np.save(save_path, est_stack)
#     # return est_stack 
#     return W.A