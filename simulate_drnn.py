import numpy as np
from drnn import DRNN
import usvt as USVT
from hyperopt import hp
from tqdm import tqdm

import warnings
np.warnings = warnings

def gendata_lin_mcar(N, T, p, seed, r = 4) : 
    """ 
    Generates data using bilinear model with uniform latent factors of dimension r
    """
    np.random.seed(seed = seed)
    ## Data Matrix (N * T)
    Data = np.zeros( (N, T) )

    # user_range = 0.9
    # time_range = 0.9

    # user_std = 0.3
    # time_std = 0.3

    U = np.random.uniform(-1, 1, size=(N,r)) #* user_range * 2 - user_range
    V = np.random.uniform(-1, 1, size=(T,r)) #* time_range * 2 - time_range
    Y = np.matmul(U, V.transpose())
    #1/np.sqrt(r) * np.matmul(U, V.transpose())

    # a = np.random.normal(0, user_std, size=N)
    # b = np.random.normal(0, time_std, size=T)
    # eps = np.random.normal(0, 0.05, size=(N, T))

    # treatment effect
    # delta(i,j) = a(i) + b(t) + eps(i,t)

    # aa = np.broadcast_to(a.reshape(N,1), (N,T))
    # bb = np.broadcast_to(b, (N,T))
    # delta = aa + bb + eps 
    # Y1 = Y0 + delta 

    #Y1 += np.random.normal(0, 0.001, size=(N,T))
    # gaussian noise
    Theta = Y
    Y += np.random.normal(0, 0.001, size=(N,T))
    
    Masking = np.zeros( (N, T) )

    Masking = np.reshape(np.random.binomial(1, p, (N*T)), (N, T))

    # Data[Masking == 1] = Y1[Masking == 1]
    # Data[Masking == 0] = Y0[Masking == 0]
    Data = Y
    return Data, Theta, Masking

def gendata_nonlin_mcar(N, T, p, seed, non_lin = "expit", r = 4):
  """
  Generates data using nonlinear model (default: expit) with uniform latent factors of dimension r
  """
  np.random.seed(seed = seed)
  #expit = lambda x : np.exp(x)/(1 + np.exp(x))
  Data = np.zeros( (N, T) )
  U = np.random.uniform(-1, 1, size=(N,r))
  V = np.random.uniform(-1, 1, size=(T,r))
  if non_lin == "expit":
    Y = (np.matmul(U, V.transpose()))**3
  else:
    raise ValueError(non_lin + " is not a supported non-linear transformation.") 
  Theta = Y
  Y += np.random.normal(0, 0.001, size=(N,T))

  Masking = np.zeros( (N, T) )

  Masking = np.reshape(np.random.binomial(1, p, (N*T)), (N, T))

  # Data[Masking == 1] = Y1[Masking == 1]
  # Data[Masking == 0] = Y0[Masking == 0]
  Data = Y
  return Data, Theta, Masking 

# def test_ssplit(nsims=30, p=0.5, r=4, k=10, model="bilinear"):
#   nsim = nsims
#   drnn_spc = {
#     'row_eta': hp.uniform('row_eta', 0, 3.5),
#     'col_eta': hp.uniform('col_eta', 0, 3.5)
#   }
#   mcar_drnn = DRNN(eta_space = hp.uniform('eta', 0, 3.5), drnn_eta_space=drnn_spc)
#   pools = []
#   pools_unit = []
#   pools_time = []
#   pools_eta_drnn = []
#   pools_eta_unit = []
#   pools_eta_time = []
#   pools_usvt = []

#   drnn_all_ests = []
#   unit_all_ests = []
#   time_all_ests = []
#   usvt_all_ests = []
#   truth_all_vals = []
#   for i, size_exp in enumerate(np.arange(4, 8)):
#     print(f"{i}-th iteration")
#     N = 2**(size_exp)
#     T = N
#     perf_drnn, eta_drnn = np.zeros(nsim), np.zeros([nsim, 4])
#     perf_unit, eta_unit = np.zeros(nsim), np.zeros([nsim, 2])
#     perf_time, eta_time = np.zeros(nsim), np.zeros([nsim, 2])
#     perf_usvt = np.zeros(nsim)
#     drnn_ests = []
#     unit_ests = []
#     time_ests = []
#     usvt_ests = []
#     truth_vals = []
#     for sim in tqdm(range(nsim)):
#       if model == "bilinear":
#         Z, Theta, M = gendata_lin_mcar(N, T, p = p, r = r, seed = sim + 42)
#       else:
#         Z, Theta, M = gendata_nonlin_mcar(N, T, p = p, r = r, non_lin = model, seed = sim + 42)
#       target_inds = [(1, 1)]
#       est_mask = np.logical_not(M)
#       est_mask[target_inds] = 1
#       est_Z = np.ma.masked_array(Z, est_mask)

#       # no ssplit distances
#       rd, cd = mcar_drnn.distances(est_Z)

#       eta_drnn_row_lr, eta_drnn_col_lr, eta_drnn_row_ur, eta_drnn_col_ur = mcar_drnn.search_eta_drnn(Z, 
#                                                         Ms, 
#                                                         fold_row_dists, 
#                                                         fold_col_dists, 
#                                                         seed = sim + 42, 
#                                                         k = k,
#                                                         max_evals=100, 
#                                                         verbose = False)


def grow_NT(nsims = 30, p = 0.5, r = 4, k = 10, ssplit=True, model = "bilinear", debug = False):
  nsim = nsims
  drnn_spc = {
    'row_eta': hp.uniform('row_eta', 0, 3.5),
    'col_eta': hp.uniform('col_eta', 0, 3.5)
  }
  mcar_drnn = DRNN(eta_space = hp.uniform('eta', 0, 3.5), drnn_eta_space=drnn_spc)
  pools = []
  pools_unit = []
  pools_time = []
  pools_eta_drnn = []
  pools_eta_unit = []
  pools_eta_time = []
  pools_usvt = []

  drnn_all_ests = []
  unit_all_ests = []
  time_all_ests = []
  usvt_all_ests = []
  truth_all_vals = []
  for i, size_exp in enumerate(np.arange(4, 8)):
    print(f"{i}-th iteration")
    N = 2**(size_exp)
    T = N
    perf_drnn, eta_drnn = np.zeros(nsim), np.zeros([nsim, 4])
    perf_unit, eta_unit = np.zeros(nsim), np.zeros([nsim, 2])
    perf_time, eta_time = np.zeros(nsim), np.zeros([nsim, 2])
    perf_usvt = np.zeros(nsim)
    drnn_ests = []
    unit_ests = []
    time_ests = []
    usvt_ests = []
    truth_vals = []
    for sim in tqdm(range(nsim)):
      if model == "bilinear":
        Z, Theta, M = gendata_lin_mcar(N, T, p = p, r = r, seed = sim + 2024)
      else:
        Z, Theta, M = gendata_nonlin_mcar(N, T, p = p, r = r, non_lin = model, seed = sim + 2024)
      M_usvt = M.copy()
      obvs_inds = np.nonzero(M == 1)
      rq_lr_obvs_inds_x =  obvs_inds[0][np.logical_and(obvs_inds[0] >= N // 2, obvs_inds[1] == T - 1)]
      rq_lr_obvs_inds_y = obvs_inds[1][np.logical_and(obvs_inds[1] == T - 1, obvs_inds[0] >= N // 2)]

      rq_ur_obvs_inds_x =  obvs_inds[0][np.logical_and(obvs_inds[0] < N // 2, obvs_inds[1] == T - 1)]
      rq_ur_obvs_inds_y = obvs_inds[1][np.logical_and(obvs_inds[1] == T - 1, obvs_inds[0] < N // 2)]

      lr_rq_inds = (rq_lr_obvs_inds_x, rq_lr_obvs_inds_y)
      ur_rq_inds = (rq_ur_obvs_inds_x, rq_ur_obvs_inds_y)

      #rq_inds = (rq_obvs_inds_x, rq_obvs_inds_y)


      # whole column for convenience   
      col_obvs_inds_x =  obvs_inds[0][obvs_inds[1] == T - 1]
      col_obvs_inds_y = obvs_inds[1][obvs_inds[1] == T - 1]
      all_rq_inds = (col_obvs_inds_x, col_obvs_inds_y)
      flattened_inds_lr = []
      flattened_inds_ur = []
      
      for b in range(len(lr_rq_inds[0])):
        flattened_inds_lr.append((lr_rq_inds[0][b], lr_rq_inds[1][b]))
      for b in range(len(ur_rq_inds[0])):
        flattened_inds_ur.append((ur_rq_inds[0][b], ur_rq_inds[1][b]))

      eval_mask_lr = np.logical_not(M)
      eval_mask_ur = np.logical_not(M)

      # M[lr_rq_inds] = 0
      # M[ur_rq_inds] = 0
      eval_mask_lr[lr_rq_inds] = 1
      eval_mask_ur[ur_rq_inds] = 1
      
      #M[col_inds] = 0
      # mask = np.logical_not(M)
      # users for lr uses ll
      lr_Z_users = np.ma.masked_array(Z[N//2:, :T//2], eval_mask_lr[N//2:, :T//2])
      lr_row_dists = mcar_drnn.distances(lr_Z_users, dist_type = "u")
      lr_rd = np.full([N, N], np.inf)
      lr_rd[N//2:, N//2:] = lr_row_dists

      # users for ur uses ul
      ur_Z_users = np.ma.masked_array(Z[:N//2, :T//2], eval_mask_ur[:N//2, :T//2])
      ur_row_dists = mcar_drnn.distances(ur_Z_users, dist_type = "u")
      ur_rd = np.full([N, N], np.inf)
      ur_rd[:N//2, :N//2] = ur_row_dists

      # fold_row_dists = ([lr_rd] * k) + ([ur_rd] * k)
      
      # cols for lr uses ur
      lr_Z_cols = np.ma.masked_array(Z[:N//2, T//2:], eval_mask_lr[:N//2, T//2:])
      lr_col_dists = mcar_drnn.distances(lr_Z_cols, dist_type = "i")
      lr_cd = np.full([T, T], np.inf)
      lr_cd[T//2:, T//2:] = lr_col_dists

      # cols for ur uses lr
      ur_Z_cols = np.ma.masked_array(Z[N//2:, T//2:], eval_mask_ur[N//2:, T//2:])
      ur_col_dists = mcar_drnn.distances(ur_Z_cols, dist_type = "i")
      ur_cd = np.full([T, T], np.inf)
      ur_cd[T//2:, T//2:] = ur_col_dists

      # fold_col_dists = ([lr_cd] * k) + ([ur_cd] * k)

      # ul users uses ur
      ul_Z_users = np.ma.masked_array(Z[:N//2, T//2:], eval_mask_lr[:N//2, T//2:])
      ul_row_dists = mcar_drnn.distances(ul_Z_users, dist_type = "u")
      ul_rd = np.full([N, N], np.inf)
      ul_rd[:N//2, :N//2] = ul_row_dists

      # users for ll uses lr
      ll_Z_users = np.ma.masked_array(Z[N//2:, T//2:], eval_mask_ur[N//2:, T//2:])
      ll_row_dists = mcar_drnn.distances(ll_Z_users, dist_type = "u")
      ll_rd = np.full([N, N], np.inf)
      ll_rd[N//2:, N//2:] = ll_row_dists

      # fold_row_dists = ([ul_rd] * k) + ([ll_rd] * k)
      fold_row_dists = [ul_rd, ll_rd]
      
      # cols for ul uses ll
      ul_Z_cols = np.ma.masked_array(Z[N//2:, :T//2], eval_mask_lr[N//2:, :T//2])
      ul_col_dists = mcar_drnn.distances(ul_Z_cols, dist_type = "i")
      ul_cd = np.full([T, T], np.inf)
      ul_cd[:T//2, :T//2] = ul_col_dists

      # cols for ll uses ul
      ll_Z_cols = np.ma.masked_array(Z[:N//2, :T//2], eval_mask_ur[:N//2, :T//2])
      ll_col_dists = mcar_drnn.distances(ll_Z_cols, dist_type = "i")
      ll_cd = np.full([T, T], np.inf)
      ll_cd[:T//2, :T//2] = ll_col_dists

      #fold_col_dists = ([ul_cd] * k / 2) + ([ll_cd] * k / 2)
      fold_col_dists = [ul_cd, ll_cd]

      M_lr = M.copy()
      M_ur = M.copy()
      M_lr[lr_rq_inds] = 0
      M_ur[ur_rq_inds] = 0
      Ms = [M_lr, M_ur]#([M_lr] * k) + ([M_ur] * k)
      M_cv = M.copy()
      M_cv[all_rq_inds] = 0

      # eta search
      eta_drnn_row_lr, eta_drnn_col_lr, eta_drnn_row_ur, eta_drnn_col_ur = mcar_drnn.search_eta_drnn(Z, 
                                                             Ms, 
                                                             fold_row_dists, 
                                                             fold_col_dists, 
                                                             seed = sim + 2024, 
                                                             k = k,
                                                             ssplit=ssplit,
                                                             max_evals=100, 
                                                             verbose = False)
      eta_star_row_lr, eta_star_row_ur = mcar_drnn.search_eta_snn(Z, 
                                              Ms, 
                                              nn_type = "u",
                                              dists = fold_row_dists,
                                              seed = sim + 2024,
                                              k = k, 
                                              ssplit=ssplit,
                                              max_evals=50, 
                                              verbose = False)
      eta_star_col_lr, eta_star_col_ur = mcar_drnn.search_eta_snn(Z, 
                                              Ms, 
                                              nn_type = "i",
                                              dists = fold_col_dists,
                                              seed = sim + 2024,
                                              k = k, 
                                              ssplit=ssplit,
                                              max_evals=50, 
                                              verbose = False)

      # mask lower right indices
      # lr_obvs_inds_x =  obvs_inds[0][np.logical_and(obvs_inds[0] >= N // 2, obvs_inds[1] >= T // 2)]
      # lr_obvs_inds_y = obvs_inds[1][np.logical_and(obvs_inds[1] >= T // 2, obvs_inds[0] >= N // 2)]
      # lr_inds = (lr_obvs_inds_x, lr_obvs_inds_y)

      # ur_obvs_inds_x =  obvs_inds[0][np.logical_and(obvs_inds[0] < N // 2, obvs_inds[1] >= T // 2)]
      # ur_obvs_inds_y = obvs_inds[1][np.logical_and(obvs_inds[1] >= T // 2, obvs_inds[0] < N // 2)]
      # ur_inds = (ur_obvs_inds_x, ur_obvs_inds_y)

      # lr_mask = np.logical_not(M)
      # lr_mask[lr_inds] = 1
      # lr_mask[ur_rq_inds] = 0
      # lr_eval_Z = np.ma.masked_array(Z, lr_mask)

      # ur_mask = np.logical_not(M)
      # ur_mask[ur_inds] = 1
      # ur_mask[lr_rq_inds] = 0
      # ur_eval_Z = np.ma.masked_array(Z, ur_mask)

      # # mse distances
      # lr_row_dists, lr_col_dists = mcar_drnn.distances(lr_eval_Z, M = None)
      # ur_row_dists, ur_col_dists = mcar_drnn.distances(ur_eval_Z, M = None)
      
      # FOR TESTING -> only lower right entries
      # true value
      #col_truth = Z[all_rq_inds]
      col_truth = Theta[all_rq_inds]
      col_truth_lr = Theta[lr_rq_inds]
      col_truth_ur = Theta[ur_rq_inds]
      #lr_truth = Z[lr_rq_inds]
      
      # ur_truth = Z[ur_rq_inds]
      
      # estimate and compute err wrt truth
      Z_est = np.ma.masked_array(Z, np.logical_not(M))
      lr_est = mcar_drnn.estimate(Z_est, M, eta_drnn_row_lr, eta_drnn_col_lr, flattened_inds_lr, lr_rd, lr_cd, debug = debug, cv = False)
      #print("______________________________________________________________")
      ur_est = mcar_drnn.estimate(Z_est, M, eta_drnn_row_ur, eta_drnn_col_ur, flattened_inds_ur, ur_rd, ur_cd, debug = debug, cv = False)
      #print("______________________________________________________________")
      lr_snn_unit_est = mcar_drnn.snn_estimate(Z_est, M, eta_star_row_lr, flattened_inds_lr, lr_rd, nn_type = "u", debug = debug, cv = False)
      #print("______________________________________________________________")
      ur_snn_unit_est = mcar_drnn.snn_estimate(Z_est, M, eta_star_row_ur, flattened_inds_ur, ur_rd, nn_type = "u", debug = debug, cv = False)
      #print("______________________________________________________________")
      lr_snn_time_est = mcar_drnn.snn_estimate(Z_est, M, eta_star_col_lr, flattened_inds_lr, lr_cd, nn_type = "i", debug = debug, cv = False)
      #print("______________________________________________________________")      
      ur_snn_time_est = mcar_drnn.snn_estimate(Z_est, M, eta_star_col_ur, flattened_inds_ur, ur_cd, nn_type = "i", debug = debug, cv = False)

      drnn_est = np.zeros([N, T])
      drnn_est[lr_rq_inds] = lr_est[lr_rq_inds]
      drnn_est[ur_rq_inds] = ur_est[ur_rq_inds]
      
      unit_est = np.zeros([N, T])
      unit_est[lr_rq_inds] = lr_snn_unit_est[lr_rq_inds]
      unit_est[ur_rq_inds] = ur_snn_unit_est[ur_rq_inds]

      time_est = np.zeros([N, T])
      time_est[lr_rq_inds] = lr_snn_time_est[lr_rq_inds]
      time_est[ur_rq_inds] = ur_snn_time_est[ur_rq_inds]

      # lr_err = mcar_drnn.avg_error(est[rq_inds], truth)
      # unit_err = mcar_drnn.avg_error(snn_unit_est[rq_inds], truth)
      # time_err = mcar_drnn.avg_error(snn_time_est[rq_inds], truth)

      drnn_err = mcar_drnn.avg_abs_error(drnn_est[all_rq_inds], col_truth)
      unit_err = mcar_drnn.avg_abs_error(unit_est[all_rq_inds], col_truth)
      time_err = mcar_drnn.avg_abs_error(time_est[all_rq_inds], col_truth)
      
      # usvt estimation, M already masks last col
      #usvt_ests_m = np.full([N, T], np.nan)
      Z[M_usvt == 0] = np.nan
      #for (i, j) in flattened_inds:
      Z_usvt = Z.copy()
      #Z_usvt[rq_inds] = np.nan
      Z_usvt[all_rq_inds] = np.nan
      m = USVT.usvt(Z_usvt)
      #usvt_ests[rq_inds] = m[i, j]



      usvt_err = mcar_drnn.avg_abs_error(m[all_rq_inds], col_truth)
      perf_drnn[sim] = drnn_err
      perf_time[sim] = time_err
      perf_unit[sim] = unit_err
      perf_usvt[sim] = usvt_err
      eta_drnn[sim] = np.array([eta_drnn_row_lr, eta_drnn_col_lr, eta_drnn_row_ur, eta_drnn_col_ur])
      eta_unit[sim] = np.array([eta_star_row_lr, eta_star_row_ur])
      eta_time[sim] = np.array([eta_star_col_lr, eta_star_col_ur])
      if eta_drnn_row_ur == eta_star_col_ur:
        raise ValueError
      drnn_ests.append(drnn_est[all_rq_inds])
      unit_ests.append(unit_est[all_rq_inds])
      time_ests.append(time_est[all_rq_inds])
      usvt_ests.append(m[all_rq_inds])
      truth_vals.append(col_truth)

    pools.append(perf_drnn)
    pools_eta_drnn.append(eta_drnn)
    pools_unit.append(perf_unit)
    pools_eta_unit.append(eta_unit)
    pools_time.append(perf_time) 
    pools_eta_time.append(eta_time)
    pools_usvt.append(perf_usvt)

    drnn_all_ests.append(drnn_ests)
    unit_all_ests.append(unit_ests)
    time_all_ests.append(time_ests)
    usvt_all_ests.append(usvt_ests)
    truth_all_vals.append(truth_vals)
  if not debug:
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_abs_err.npy", np.array(pools))
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_abs_eta.npy", np.array(pools_eta_drnn))
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_unit_abs_err.npy", np.array(pools_unit))
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_unit_abs_eta.npy", np.array(pools_eta_unit))
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_time_abs_err.npy", np.array(pools_time))
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_time_abs_eta.npy", np.array(pools_eta_time))
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_usvt_abs_err.npy", np.array(pools_usvt))

    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_estimates.npy", np.array(drnn_all_ests, dtype = object), allow_pickle=True)
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_unit_estimates.npy", np.array(unit_all_ests, dtype = object), allow_pickle=True)
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_time_estimates.npy", np.array(time_all_ests, dtype = object), allow_pickle=True)
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_usvt_estimates.npy", np.array(usvt_all_ests, dtype = object), allow_pickle=True)
    np.save("drnn_full_bugll_ssplit_mcar_N4_7_blin_uv11_d4_p5_truth_vals.npy", np.array(truth_all_vals, dtype = object), allow_pickle=True)

def fix_NT(size = 2**7, p = 0.5, model = "bilinear", r = 4, sim_seed = 42):
  sim_seed = sim_seed
  size = size
  N, T = size, size
  drnn_spc = {
    'row_eta': hp.uniform('row_eta', 0, 1.5),
    'col_eta': hp.uniform('col_eta', 0, 1.5)
  }
  mcar_drnn = DRNN(eta_space = hp.uniform('eta', 0, 1.5), drnn_eta_space=drnn_spc)
  if model == "bilinear":
    Z, M = gendata_lin_mcar(N, T, p = p, r = r, seed = sim_seed)
  else:
    Z, M = gendata_nonlin_mcar(N, T, p = p, r = r, non_lin = model, seed = sim_seed)

  M_usvt = M.copy()
  obvs_inds = np.nonzero(M == 1)
  rq_lr_obvs_inds_x =  obvs_inds[0][np.logical_and(obvs_inds[0] >= N // 2, obvs_inds[1] == T - 1)]
  rq_lr_obvs_inds_y = obvs_inds[1][np.logical_and(obvs_inds[1] == T - 1, obvs_inds[0] >= N // 2)]

  rq_ur_obvs_inds_x =  obvs_inds[0][np.logical_and(obvs_inds[0] < N // 2, obvs_inds[1] == T - 1)]
  rq_ur_obvs_inds_y = obvs_inds[1][np.logical_and(obvs_inds[1] == T - 1, obvs_inds[0] < N // 2)]

  lr_rq_inds = (rq_lr_obvs_inds_x, rq_lr_obvs_inds_y)
  ur_rq_inds = (rq_ur_obvs_inds_x, rq_ur_obvs_inds_y)

  # whole column for convenience   
  col_obvs_inds_x =  obvs_inds[0][obvs_inds[1] == T - 1]
  col_obvs_inds_y = obvs_inds[1][obvs_inds[1] == T - 1]
  all_rq_inds = (col_obvs_inds_x, col_obvs_inds_y)
  flattened_inds_lr = []
  flattened_inds_ur = []
  
  #col_inds = (col_obvs_inds_x, col_obvs_inds_y)
  # rq_inds = [(N-1, T-1)]
  for b in range(len(ur_rq_inds[0])):
    flattened_inds_ur.append((ur_rq_inds[0][b], ur_rq_inds[1][b]))
  for b in range(len(lr_rq_inds[0])):
    flattened_inds_lr.append((lr_rq_inds[0][b], lr_rq_inds[1][b]))

  M[ur_rq_inds] = 0
  M[lr_rq_inds] = 0
  eval_mask = np.logical_not(M)

  masked_Z = np.ma.masked_array(Z, eval_mask)

  # eta search
  eta_drnn_row, eta_drnn_col = mcar_drnn.search_eta_drnn(masked_Z, M, max_evals=100, verbose = True)
  eta_star_row = mcar_drnn.search_eta_snn(masked_Z, M, nn_type = "u", max_evals=100, verbose = True)
  eta_star_col = mcar_drnn.search_eta_snn(masked_Z, M, nn_type = "i", max_evals=100, verbose = True)

  # mask lower right indices
  lr_obvs_inds_x =  obvs_inds[0][np.logical_and(obvs_inds[0] >= N // 2, obvs_inds[1] >= T // 2)]
  lr_obvs_inds_y = obvs_inds[1][np.logical_and(obvs_inds[1] >= T // 2, obvs_inds[0] >= N // 2)]
  lr_inds = (lr_obvs_inds_x, lr_obvs_inds_y)

  ur_obvs_inds_x =  obvs_inds[0][np.logical_and(obvs_inds[0] < N // 2, obvs_inds[1] >= T // 2)]
  ur_obvs_inds_y = obvs_inds[1][np.logical_and(obvs_inds[1] >= T // 2, obvs_inds[0] < N // 2)]
  ur_inds = (ur_obvs_inds_x, ur_obvs_inds_y)

  lr_mask = np.logical_not(M)
  lr_mask[lr_inds] = 1
  lr_mask[ur_rq_inds] = 0
  lr_eval_Z = np.ma.masked_array(Z, lr_mask)

  ur_mask = np.logical_not(M)
  ur_mask[ur_inds] = 1
  ur_mask[lr_rq_inds] = 0
  ur_eval_Z = np.ma.masked_array(Z, ur_mask)

  # mse distances
  lr_row_dists, lr_col_dists = mcar_drnn.distances(lr_eval_Z, M = None)
  ur_row_dists, ur_col_dists = mcar_drnn.distances(ur_eval_Z, M = None)
  # true value
  col_truth = Z[all_rq_inds]
  # lr_truth = Z[lr_rq_inds]
  # ur_truth = Z[ur_rq_inds]

  # estimate and compute err wrt truth
  lr_est = mcar_drnn.estimate(lr_eval_Z, M, eta_drnn_row, eta_drnn_col, flattened_inds_lr, lr_row_dists, lr_col_dists, cv = True)
  ur_est = mcar_drnn.estimate(ur_eval_Z, M, eta_drnn_row, eta_drnn_col, flattened_inds_ur, ur_row_dists, ur_col_dists, cv = True)

  lr_snn_unit_est = mcar_drnn.snn_estimate(lr_eval_Z, M, eta_star_row, flattened_inds_lr, lr_row_dists, nn_type = "u", cv = True)
  ur_snn_unit_est = mcar_drnn.snn_estimate(ur_eval_Z, M, eta_star_row, flattened_inds_ur, ur_row_dists, nn_type = "u", cv = True)

  lr_snn_time_est = mcar_drnn.snn_estimate(lr_eval_Z, M, eta_star_col, flattened_inds_lr, lr_col_dists, nn_type = "i", cv = True)
  ur_snn_time_est = mcar_drnn.snn_estimate(ur_eval_Z, M, eta_star_col, flattened_inds_ur, ur_col_dists, nn_type = "i", cv = True)

  drnn_est = np.zeros([N, T])
  drnn_est[lr_rq_inds] = lr_est[lr_rq_inds]
  drnn_est[ur_rq_inds] = ur_est[ur_rq_inds]
  
  unit_est = np.zeros([N, T])
  unit_est[lr_rq_inds] = lr_snn_unit_est[lr_rq_inds]
  unit_est[ur_rq_inds] = ur_snn_unit_est[ur_rq_inds]

  time_est = np.zeros([N, T])
  time_est[lr_rq_inds] = lr_snn_time_est[lr_rq_inds]
  time_est[ur_rq_inds] = ur_snn_time_est[ur_rq_inds]

  # lr_err = mcar_drnn.avg_error(est[rq_inds], truth)
  # unit_err = mcar_drnn.avg_error(snn_unit_est[rq_inds], truth)
  # time_err = mcar_drnn.avg_error(snn_time_est[rq_inds], truth)

  drnn_errs = mcar_drnn.all_error(drnn_est[all_rq_inds], col_truth)
  unit_errs = mcar_drnn.all_error(unit_est[all_rq_inds], col_truth)
  time_errs = mcar_drnn.all_error(time_est[all_rq_inds], col_truth)
  
  # usvt estimation, M already masks last col
  usvt_ests = np.full([N, T], np.nan)
  Z[M_usvt == 0] = np.nan
  for (i, j) in flattened_inds_ur:
    Z_usvt = Z.copy()
    #Z_usvt[rq_inds] = np.nan
    Z_usvt[i, j] = np.nan
    #Z_usvt[all_rq_inds] = np.nan
    m = USVT.usvt(Z_usvt)
    usvt_ests[i, j] = m[i, j]
  for (i, j) in flattened_inds_lr:
    Z_usvt = Z.copy()
    #Z_usvt[rq_inds] = np.nan
    Z_usvt[i, j] = np.nan
    #Z_usvt[all_rq_inds] = np.nan
    m = USVT.usvt(Z_usvt)
    usvt_ests[i, j] = m[i, j]

  usvt_errs = mcar_drnn.all_error(usvt_ests[all_rq_inds], col_truth)
  
  np.save("drnn_mcar_N128_abs_drnn_errors.npy", drnn_errs)
  np.save("drnn_mcar_N128_abs_unit_errors.npy", unit_errs)
  np.save("drnn_mcar_N128_abs_time_errors.npy", time_errs)
  np.save("drnn_mcar_N128_tryabs_usvt_errors.npy", usvt_errs)

def main():
  grow_NT(nsims = 5, p = 0.5, debug = False)
  
if __name__ == "__main__":
    main()



          
