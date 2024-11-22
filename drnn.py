'''
Implement Doubly Robust Nearest Neighbors on top of the NNImputer frameowrk
introduced for distributional nearest neighbors
'''

from nnimputer import NNImputer
import numpy as np
from hyperopt import atpe, tpe, hp, Trials, fmin, STATUS_OK
from tqdm import tqdm
from matplotlib import pyplot as plt

class DRNN(NNImputer):
  def __init__(
    self,
    eta_axis=0,
    eta_space=None,
    drnn_eta_space = None,
    search_algo=tpe.suggest,
    k=None,
    rand_seed=None
    ):
    """
    Parameters:
    -----------
    eta_axis : integer in [0, 1].
              Indicates which axis to compute the eta search over. If eta search is
              done via blocks (i.e. not row-wise or column-wise), then this parameter is ignored
    eta_space : a hyperopt hp search space
                for example: hp.uniform('eta', 0, 1). If no eta_space is given,
                then this example will be the default search space.
    search_algo : a hyperopt algorithm
                  for example: tpe.suggest, default is tpe.suggest.
    k : integer > 1, the number of folds in k-fold cross validation over.
        If k = None (default), the LOOCV is used. 
    rand_seed : the random seed to be used for reproducible results. 
                If None is used (default), then the system time is used (not reproducible)
    """
    super().__init__(
          nn_type = "dr",
          eta_axis = eta_axis,
          eta_space = eta_space,
          search_algo = search_algo,
          k = k,
          rand_seed = rand_seed
      )
    self.drnn_eta_space = drnn_eta_space

  def distances(self, Z_masked, M = None, i = 0, t = 0, dist_type = "all"):
      """
      Compute the row/column-wise MSE distance

      Parameters:
      -----------
      Z : a (masked) matrix of size N x T

      Arguments:
      ----------
      """
      N, T = Z_masked.shape
      # note: should adjust for more than 1 treatment in mask
      #Z_masked = np.ma.masked_array(Z, mask = M != 1)
      
      if dist_type == "all":
          col_Z = np.swapaxes(Z_masked, 0, 1)
          Z_br_col = col_Z[:, None]
          Z_br_row = Z_masked[:, None]
          row_dists = np.mean((Z_br_row - Z_masked)**2, axis = 2)
          col_dists = np.mean((Z_br_col - col_Z)**2, axis = 2)
          np.fill_diagonal(row_dists, np.inf)
          np.fill_diagonal(col_dists, np.inf)
          return (row_dists.filled(np.nan), col_dists.filled(np.nan))
      elif dist_type == "single entry":
          entry_row = Z_masked[i]
          entry_col = Z_masked[:, t]
          row_dists = np.mean((entry_row - Z_masked)**2, axis = 1)
          col_dists = np.mean((entry_col - np.swapaxes(Z_masked, 0, 1))**2, axis = 1)
          row_dists[i] = np.inf
          col_dists[i] = np.inf
          return row_dists.filled(np.nan), col_dists.filled(np.nan)
      elif dist_type == "u":
        Z_br_row = Z_masked[:, None]
        row_dists = np.mean((Z_br_row - Z_masked)**2, axis = 2)
        np.fill_diagonal(row_dists, np.inf)
        return row_dists.filled(np.nan)
      elif dist_type == "i":
        col_Z = np.swapaxes(Z_masked, 0, 1)
        Z_br_col = col_Z[:, None]
        col_dists = np.mean((Z_br_col - col_Z)**2, axis = 2)
        np.fill_diagonal(col_dists, np.inf)
        return col_dists.filled(np.nan)
      
  def estimate(self, Z, M, eta_rows, eta_cols, inds, row_dists, col_dists, cv = True, debug = False):
    """
    Return the DRNN estimate for inds

    Parameters:
    -----------
    Z : a (masked) matrix of size N x T
    """
    N, T = Z.shape
    s_time_nn_all = col_dists <= eta_cols  # Shape (T, T)
    s_unit_nn_all = row_dists <= eta_rows  # Shape (N, N)
    ests = np.full([N, T], np.nan)
    for i, j in inds:
      s_unit_nn  = s_unit_nn_all[i]
      if debug:
        print(row_dists[i])
        print("Row Eta: " + str(eta_rows))
        print("DRNN Unit: " + str(i) + ", " + str(np.nonzero(s_unit_nn)))
      s_time_nn = s_time_nn_all[j]
      if debug:
        print(col_dists[j])
        print("Col Eta: " + str(eta_cols))
        print("DRNN Time: " + str(j) + ", " + str(np.nonzero(s_time_nn)))

      #print(s_unit_nn.shape)
      row_Ys = Z[s_unit_nn, j][:, np.newaxis]
      col_Ys = Z[i, s_time_nn]
      #print(len(col_Ys))
      est = np.nan
      if len(row_Ys) > 0 and len(col_Ys) > 0:
        est = np.mean(row_Ys + col_Ys - Z[s_unit_nn][:, s_time_nn])
      #  print(s_unit_nn)
      #  print(s_time_nn)
      #  print(est)
      if est is np.ma.masked or np.isnan(est):
          #est = Z.data[i, j]
      #print("No neighbors in drnn estimate")
        full_row_Ys = Z[:, j][:, np.newaxis]
        full_col_Ys = Z[i, :]
        all = np.mean(full_row_Ys + full_col_Ys - Z)
        est = all if all is not np.ma.masked and ~np.isnan(all) else 0
      ests[i, j] = est
    return ests
  
  def snn_estimate(self, Z, M, eta, inds, dists, nn_type, cv = True, debug = False):
    """
    Compute user-user or item-item nearest neigbors
    """
    N, T = Z.shape
    ests = np.full([N, T], np.nan)
    if nn_type == "u":
      for i, j in inds:
        nn_inds = dists[i] <= eta
        if debug:
          print(dists[i])
          print("Row Eta: " + str(eta))
          print("Unit: " + str(i) + ", " + str(np.nonzero(nn_inds)))
        est = np.mean(Z[nn_inds, j])
        if est is np.ma.masked or np.isnan(est):
            alt_est = np.mean(Z[:, j])
            # note: if no nn, then avg over whole col
            ests[i, j] = alt_est if alt_est is not np.ma.masked and ~np.isnan(alt_est) else 0
        else:
            ests[i, j] = est 
    elif nn_type == "i": 
      for i, j in inds:
        nn_inds = dists[j] <= eta 
        if debug:
          print(dists[j])
          print("Col Eta: " + str(eta))
          print("Time: " + str(j) + ", " + str(np.nonzero(nn_inds)))
        est = np.mean(Z[i, nn_inds])
        if est is np.ma.masked or np.isnan(est):
          alt_est = np.mean(Z[i, :])
          # note: if no nn, then avg over whole row
          ests[i, j] = alt_est if alt_est is not np.ma.masked and ~np.isnan(alt_est) else 0
        else:
          ests[i, j] = est
    else:
      raise ValueError("Invalid nearest neighbors type " + str(self.nn_type))
    return ests
  
  def cv_drnn_square(self, Z, M, folds, row_dists, col_dists, row_eta, col_eta, ssplit=True):
    """
    Tuning etas for drnn with upper right and lower right quadrant folds
    """
    tot_err = 0
    k = len(folds)
    all_errs = np.full([k], np.nan)
    #print("In cv drnn")
    for i, ho_inds in enumerate(folds):
      # M = Ms[i % 2]
      M = M if ssplit else M[i]
      cv_mask = np.logical_not(M)
      cv_mask[ho_inds] = 1
      cv_Z = np.ma.masked_array(Z, cv_mask)
      row_dist = row_dists if ssplit else row_dists[i]
      col_dist = col_dists if ssplit else col_dists[i]
      flattened_inds = []
      for b in range(len(ho_inds[0])):
        flattened_inds.append((ho_inds[0][b], ho_inds[1][b]))
      estimate = self.estimate(cv_Z, M, row_eta, col_eta, flattened_inds, row_dist, col_dist)
      truth = Z[ho_inds]
      all_errs[i] = self.avg_error(estimate[ho_inds], truth)
    
    return np.nanmean(all_errs)

  def cv_snn_square(self, Z, M, folds, dists, nn_type, eta, ssplit=True):
    """
    Tuning eta for nn_type nearest neighbors
    """
    k = len(folds)
    tot_err = 0
    all_errs = np.full([k], np.nan)
    for i, ho_inds in enumerate(folds):
      M = M if ssplit else M[i]
      cv_mask = np.logical_not(M)
      cv_mask[ho_inds] = 1
      cv_Z = np.ma.masked_array(Z, cv_mask)
      dist = dists if ssplit else dists[i]
      flattened_inds = []
      for b in range(len(ho_inds[0])):
        flattened_inds.append((ho_inds[0][b], ho_inds[1][b]))
      estimate = self.snn_estimate(cv_Z, M, eta, flattened_inds, dist, nn_type = nn_type)
      truth = Z[ho_inds]
      all_errs[i] = self.avg_error(estimate[ho_inds], truth)

    return np.nanmean(all_errs)
  
  def cv_drnn_full(self, Z, M, seed, k = 5, max_evals = 200, verbose = True):
    np.random.seed(seed=seed)
    obvs_inds = np.nonzero(M == 1)
    obvs_inds_x = obvs_inds[0]
    obvs_inds_y = obvs_inds[1]
    rand_inds = np.arange(len(obvs_inds_x))
    np.random.shuffle(rand_inds)
    fold_inds =  np.array_split(rand_inds, k)
    folds = [(obvs_inds_x[inds], obvs_inds_y[inds]) for inds in fold_inds]

    # compute differences for each fold
    r_dists = []
    c_dists = []
    Ms = []
    for f in folds:
      cv_M = M.copy()
      cv_M[f] = 0
      Ms.append(cv_M)
      cv_Z = np.ma.masked_array(Z, np.logical_not(cv_M))
      rd, cd = self.distances(cv_Z, dist_type = "all")
      r_dists.append(rd)
      c_dists.append(cd)
    
    def obj(params):
      loss = self.cv_drnn_square(Z, Ms, folds, r_dists, c_dists, params["row_eta"], params["col_eta"], ssplit=False)
      #print(type(loss))
      return loss #{"loss" : loss, "status" : STATUS_OK}

    trials = Trials()
    best_eta = fmin(
      fn=obj,
      verbose=verbose,
      space=self.drnn_eta_space,
      algo=self.search_algo,
      max_evals=max_evals,
      trials=trials,
    )
    return best_eta["row_eta"], best_eta["col_eta"]

  def cv_drnn_ssplit(self, Z, Ms, row_dists, col_dists, seed, k = 5, max_evals = 200, verbose = True):
    np.random.seed(seed=seed)
    N, T = Z.shape
    # obvs_inds_lr = np.nonzero(Ms[0] == 1)
    # lr_obvs_inds_x = obvs_inds_lr[0][np.logical_and(obvs_inds_lr[0] >= N // 2, obvs_inds_lr[1] >= T // 2)]
    # lr_obvs_inds_y = obvs_inds_lr[1][np.logical_and(obvs_inds_lr[1] >= T // 2, obvs_inds_lr[0] >= N // 2)]
    
    # obvs_inds_ur = np.nonzero(Ms[-1] == 1)
    # ur_obvs_inds_x = obvs_inds_ur[0][np.logical_and(obvs_inds_ur[0] >= T // 2, obvs_inds_ur[1] < N // 2)]
    # ur_obvs_inds_y = obvs_inds_ur[1][np.logical_and(obvs_inds_ur[1] < N // 2, obvs_inds_ur[0] >= T // 2)]

    # FULL CV SPLIT ===================================
    obvs_inds_ul = np.nonzero(Ms[0] == 1)
    ul_obvs_inds_x = obvs_inds_ul[0][np.logical_and(obvs_inds_ul[1] < T // 2, obvs_inds_ul[0] < N // 2)]
    ul_obvs_inds_y = obvs_inds_ul[1][np.logical_and(obvs_inds_ul[0] < N // 2, obvs_inds_ul[1] < T // 2)]
    ul_inds = (ul_obvs_inds_x, ul_obvs_inds_y)
    # lr_inds = (lr_obvs_inds_x, lr_obvs_inds_y)

    obvs_inds_ll = np.nonzero(Ms[1] == 1)
    ll_obvs_inds_x = obvs_inds_ll[0][np.logical_and(obvs_inds_ll[1] < T // 2, obvs_inds_ll[0] >= N // 2)]
    ll_obvs_inds_y = obvs_inds_ll[1][np.logical_and(obvs_inds_ll[0] >= N // 2, obvs_inds_ll[1] < T // 2)]

    # five fold cv over obvs inds in ul
    rand_inds = np.arange(len(ul_obvs_inds_x))
    np.random.shuffle(rand_inds)
    split_inds = np.array_split(rand_inds, k)
    ul_folds = [(ul_obvs_inds_x[inds], ul_obvs_inds_y[inds]) for inds in split_inds]

    ll_rand_inds = np.arange(len(ll_obvs_inds_x))
    np.random.shuffle(ll_rand_inds)
    ll_split_inds = np.array_split(ll_rand_inds, k)
    ll_folds = [(ll_obvs_inds_x[inds], ll_obvs_inds_y[inds]) for inds in ll_split_inds]

    def obj_ll(params):
      loss = self.cv_drnn_square(Z, Ms[1], ll_folds, row_dists[1], col_dists[1], params["row_eta"], params["col_eta"], ssplit=True)
      #print(type(loss))
      return loss #{"loss" : loss, "status" : STATUS_OK}
    def obj_ul(params):
      loss = self.cv_drnn_square(Z, Ms[0], ul_folds, row_dists[0], col_dists[0], params["row_eta"], params["col_eta"], ssplit=True)
      #print(type(loss))
      return loss
    # ==========================
    # percentiles = np.array([5, 10, 13, 17, 20, 35, 50, 65, 80, 100])
    # eta_cand_ul_row = np.quantile(row_dists[0][np.logical_and(~np.isinf(row_dists[0]), ~np.isnan(row_dists[0]))], q = percentiles / 100)
    # eta_cand_ul_col = np.quantile(col_dists[0][np.logical_and(~np.isinf(col_dists[0]), ~np.isnan(col_dists[0]))], q = percentiles / 100)
    
    # eta_cand_ll_row = np.quantile(row_dists[1][np.logical_and(~np.isinf(row_dists[1]), ~np.isnan(row_dists[1]))], q = percentiles / 100)
    # eta_cand_ll_col = np.quantile(col_dists[1][np.logical_and(~np.isinf(col_dists[1]), ~np.isnan(col_dists[1]))], q = percentiles / 100)
 
    # #eta_cand = np.append(np.arange(0, 0.4, 0.05), np.arange(0.4, 1, 0.1)) # used for MCAR
    # perf = np.zeros([len(eta_cand_ul_row), len(eta_cand_ul_col)])
    # it_count = 0
    # for i, eta_row in enumerate(eta_cand_ul_row):
    #   eta_c_start, eta_c_end = max(0, i - 3), min(i + 3, len(eta_cand_ul_row))
    #   for j, eta_col in enumerate(eta_cand_ul_col[eta_c_start:eta_c_end]):
    #     # print(str(it_count) + "/" + str(len(eta_cand)**2))
    #     #it_count += 1
    #     perf[i, j] = self.cv_drnn_square(Z, Ms[0], ul_folds, row_dists[0], col_dists[0], eta_row, eta_col)

    #     # print("Loss: " + str(perf[i, j]))
    # r, c = np.unravel_index(np.argmin(perf, axis=None), perf.shape)
    # best_eta_ul = {"row_eta" : eta_cand_ul_row[r], "col_eta" : eta_cand_ul_col[c]}

    # perf_ll = np.zeros([len(eta_cand_ll_row), len(eta_cand_ll_col)])
    # it_count = 0
    # for i, eta_row in enumerate(eta_cand_ll_row):
    #   eta_c_start, eta_c_end = [max(0, i - 3), min(i + 3, len(eta_cand_ll_row))]
    #   for j, eta_col in enumerate(eta_cand_ll_col[eta_c_start:eta_c_end]):
    #     # print(str(it_count) + "/" + str(len(eta_cand)**2))
    #     #it_count += 1
    #     perf_ll[i, j] = self.cv_drnn_square(Z, Ms[1], ll_folds, row_dists[1], col_dists[1], eta_row, eta_col)

    #     # print("Loss: " + str(perf[i, j]))
    # r, c = np.unravel_index(np.argmin(perf_ll, axis=None), perf_ll.shape)
    # best_eta_ll = {"row_eta" : eta_cand_ll_row[r], "col_eta" : eta_cand_ll_col[c]}


    trials_ll = Trials()
    best_eta_ll = fmin(
      fn=obj_ll,
      verbose=verbose,
      space=self.drnn_eta_space,
      algo=self.search_algo,
      max_evals=max_evals,
      trials=trials_ll,
      rstate=np.random.default_rng(seed + N)
    )

    trials_ul = Trials()
    best_eta_ul = fmin(
      fn=obj_ul,
      verbose=verbose,
      space=self.drnn_eta_space,
      algo=self.search_algo,
      max_evals=max_evals,
      trials=trials_ul,
      rstate=np.random.default_rng(seed + N)
    )

    return best_eta_ul["row_eta"], best_eta_ul["col_eta"], best_eta_ll["row_eta"], best_eta_ll["col_eta"]

  def search_eta_drnn(self, Z, Ms, row_dists, col_dists, seed, k = 5, ssplit=True, max_evals = 200, verbose = True):
    """
    Tune optimal etas
    """
    if ssplit:
      return self.cv_drnn_ssplit(Z, Ms, row_dists, col_dists, seed, k, max_evals, verbose)
    else:
      return self.cv_drnn_full(Z, Ms, seed, k, max_evals, verbose)
    
    

  def search_eta_snn(self, Z, Ms, nn_type, dists, seed, ssplit=True, k = 5, max_evals=200, verbose=True):
    """
    Tune optimal eta
    """
    # # FULL CV SPLIT ==================================
    np.random.seed(seed=seed)
    N, T = Z.shape
    if ssplit:
      obvs_inds_ul = np.nonzero(Ms[0] == 1)
      ul_obvs_inds_x = obvs_inds_ul[0][np.logical_and(obvs_inds_ul[1] < T // 2, obvs_inds_ul[0] < N // 2)]
      ul_obvs_inds_y = obvs_inds_ul[1][np.logical_and(obvs_inds_ul[0] < N // 2, obvs_inds_ul[1] < T // 2)]
      # ul_inds = (ul_obvs_inds_x, ul_obvs_inds_y)
      # lr_inds = (lr_obvs_inds_x, lr_obvs_inds_y)

      obvs_inds_ll = np.nonzero(Ms[1] == 1)
      ll_obvs_inds_x = obvs_inds_ll[0][np.logical_and(obvs_inds_ll[1] < T // 2, obvs_inds_ll[0] >= N // 2)]
      ll_obvs_inds_y = obvs_inds_ll[1][np.logical_and(obvs_inds_ll[0] >= N // 2, obvs_inds_ll[1] < T // 2)]

      # five fold cv over obvs inds in ul
      rand_inds = np.arange(len(ul_obvs_inds_x))
      np.random.shuffle(rand_inds)
      split_inds = np.array_split(rand_inds, k)
      ul_folds = [(ul_obvs_inds_x[inds], ul_obvs_inds_y[inds]) for inds in split_inds]

      ll_rand_inds = np.arange(len(ll_obvs_inds_x))
      np.random.shuffle(ll_rand_inds)
      ll_split_inds = np.array_split(ll_rand_inds, k)
      ll_folds = [(ll_obvs_inds_x[inds], ll_obvs_inds_y[inds]) for inds in ll_split_inds]

      def obj_ul(eta):
        return self.cv_snn_square(Z, Ms[0], ul_folds, dists[0], nn_type, eta, ssplit=True)
      
      def obj_ll(eta):
        return self.cv_snn_square(Z, Ms[1], ll_folds, dists[1], nn_type, eta, ssplit=True)
      
      trials_ul = Trials()
      best_eta_ul = fmin(
        fn=obj_ul,
        verbose=verbose,
        space=self.eta_space,
        algo=self.search_algo,
        max_evals=max_evals,
        trials=trials_ul,
        rstate=np.random.default_rng(seed + N)
      )

      trials_ll = Trials()
      best_eta_ll = fmin(
        fn=obj_ll,
        verbose=verbose,
        space=self.eta_space,
        algo=self.search_algo,
        max_evals=max_evals,
        trials=trials_ll,
        rstate=np.random.default_rng(seed + N)
        )
      return best_eta_ul["eta"], best_eta_ll["eta"]
    else:
      obvs_inds = np.nonzero(Ms == 1)
      obvs_inds_x = obvs_inds[0]
      obvs_inds_y = obvs_inds[1]
      rand_inds = np.arange(len(obvs_inds_x))
      np.random.shuffle(rand_inds)
      fold_inds =  np.array_split(rand_inds, k)
      folds = [(obvs_inds_x[inds], obvs_inds_y[inds]) for inds in fold_inds]

      # compute differences for each fold
      dists = []
      Ms_fold = []
      for f in folds:
        cv_M = Ms.copy()
        cv_M[f] = 0
        Ms_fold.append(cv_M)
        cv_Z = np.ma.masked_array(Z, np.logical_not(cv_M))
        d = self.distances(cv_Z, dist_type = nn_type)
        dists.append(d)
      
      def obj(eta):
        return self.cv_snn_square(Z, Ms_fold, folds, dists, nn_type, eta, ssplit=False)
      
      trials = Trials()
      best_eta = fmin(
        fn=obj,
        verbose=verbose,
        space=self.eta_space,
        algo=self.search_algo,
        max_evals=max_evals,
        trials=trials,
      )
      return best_eta['eta']

    # ======================================
   
    # ========================================
    #eta_cand = np.append(np.arange(0, 0.4, 0.05), np.arange(0.3, 1, 0.1)) # used for MCAR
    # percentiles = np.array([5, 10, 13, 17, 20, 35, 50, 65, 80, 100])
    # eta_cand_ul = np.quantile(dists[0][np.logical_and(~np.isinf(dists[0]), ~np.isnan(dists[0]))], q = percentiles / 100)
    # perf = np.zeros([len(eta_cand_ul)])
    # it_count = 0
    # for i, eta in enumerate(eta_cand_ul):
    #     # print(str(it_count) + "/" + str(len(eta_cand)**2))
    #     #it_count += 1
    #     perf[i] = self.cv_snn_square(Z, Ms[0], ul_folds, dists[0], nn_type, eta)

    #     # print("Loss: " + str(perf[i, j]))
    # # r, c = np.unravel_index(np.argmin(perf, axis=None), perf.shape)
    # best = np.argmin(perf)
    # best_eta_ul = {"eta" : eta_cand_ul[best]}

    # eta_cand_ll = np.quantile(dists[1][np.logical_and(~np.isinf(dists[1]), ~np.isnan(dists[1]))], q = percentiles / 100)
    # perf_ll = np.zeros([len(eta_cand_ll)])
    # for i, eta in enumerate(eta_cand_ll):
    # # print(str(it_count) + "/" + str(len(eta_cand)**2))
    # #it_count += 1
    #   perf_ll[i] = self.cv_snn_square(Z, Ms[1], ll_folds, dists[1], nn_type, eta)

    #     # print("Loss: " + str(perf[i, j]))
    # # r, c = np.unravel_index(np.argmin(perf, axis=None), perf.shape)
    # best = np.argmin(perf_ll)
    # best_eta_ll = {"eta" : eta_cand_ll[best]}

    

  def avg_error(self, est, truth):
     return np.mean((est - truth)**2)

  def avg_abs_error(self, est, truth):
    return np.mean(np.abs(est - truth))
  
  def all_abs_error(self, est, truth):
    return np.abs(est - truth)

  def all_error(self, est, truth):
    return (est - truth)**2

