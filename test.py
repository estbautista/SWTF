import os
import yaml
import pickle
import torch
import math
import argparse
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader

#####################################################
# Helper imports
#####################################################
from data import WindowDataset, collate_single
from config import get_config, set_dir

#####################################################
# Helper functions 
#####################################################
def topk_eigvecs_gram_sparse_coo(idx, vals, shape, mode, k, which="LA"):
    """
    Computes top-k eigenpairs of Gram = Xd.T Xd without forming Xd.

    This avoids building CSR unfoldings. CSR needs an indptr array with one
    entry per row, which can be huge for ultra sparse SWTF tensors.

    This also compresses the unfolding columns. For SWTF, mode 0 has N**2
    possible columns, so even ARPACK vectors of length N**2 are too large.
    Since zero columns have zero energy, the useful eigenspace lives only on
    the active columns observed in the sparse tensor.
    """
    I1, I2, I3 = shape

    # Select unfolding columns and row keys
    if mode == 0:
        col_key = idx[0]
        row_key = idx[1] * I3 + idx[2]
    elif mode == 1:
        col_key = idx[1]
        row_key = idx[0] * I3 + idx[2]
    elif mode == 2:
        col_key = idx[2]
        row_key = idx[0] * I2 + idx[1]
    else:
        raise ValueError("mode must be 0, 1, or 2")

    # Compress only the actually used unfolding columns and rows
    active_cols, col = np.unique(col_key, return_inverse=True)
    _, row = np.unique(row_key, return_inverse=True)

    N = int(active_cols.shape[0])
    J = int(row.max()) + 1

    # Handle degenerate cases
    if N == 0:
        return np.array([]), np.empty((0, 0)), col
    if N == 1:
        return np.array([float((vals * vals).sum())]), np.ones((1, 1)), col

    # eigsh requires k < N
    k = min(k, N - 1)
    if k <= 0:
        return np.array([]), np.empty((N, 0)), col

    def matvec(v):
        # y = Xd @ v, accumulated over used rows only
        y = np.bincount(row, weights=vals * v[col], minlength=J)

        # z = Xd.T @ y, accumulated over used columns only
        z = np.bincount(col, weights=vals * y[row], minlength=N)
        return z.astype(vals.dtype, copy=False)

    # ARPACK is safer with float64, and sparse binary data is usually tiny
    dtype = np.float64
    Gop = LinearOperator((N, N), matvec=matvec, dtype=dtype)
    evals, evecs = eigsh(Gop, k=k, which=which)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order], col


def hosvd_sparse(X_sparse, ranks):
    """
    Compute HOSVD factors and core for sparse 3D tensor
    """
    r1, r2, r3 = ranks
    X_sparse = X_sparse.coalesce()
    I1, I2, I3 = X_sparse.shape

    # Get coords once
    idx_t = X_sparse.indices()
    vals_t = X_sparse.values()

    # Move coords to NumPy for SciPy LinearOperator
    idx = idx_t.cpu().numpy()
    vals = vals_t.detach().cpu().numpy().astype(np.float64, copy=False)

    # HOSVD factors via eigenvectors of Gram matrices
    # Here the Gram matvecs are computed directly from sparse COO coords
    # and both rows and columns are compressed to active sparse indices
    ev1, U1, c1 = topk_eigvecs_gram_sparse_coo(idx, vals, (I1, I2, I3), mode=0, k=r1)
    ev2, U2, c2 = topk_eigvecs_gram_sparse_coo(idx, vals, (I1, I2, I3), mode=1, k=r2)
    ev3, U3, c3 = topk_eigvecs_gram_sparse_coo(idx, vals, (I1, I2, I3), mode=2, k=r3)

    # Compute core
    # We only need factor rows touched by nonzero entries
    dtype = vals_t.dtype
    device = vals_t.device
    A = torch.as_tensor(U1[c1, :], dtype=dtype, device=device)
    B = torch.as_tensor(U2[c2, :], dtype=dtype, device=device)
    C = torch.as_tensor(U3[c3, :], dtype=dtype, device=device)
    core = torch.einsum('n,na,nb,nc->abc', vals_t, A, B, C)

    return (ev1, U1), (ev2, U2), (ev3, U3), core

def swtf_scoring(adj_tensor, ranks): 
    """
    In the paper the authors consider (A - A x_k U_k U_k.T)
    They treat mode projections individually to classify anomalies by type
    Here we only care about anomaly score.

    We can thus interpret the method as DTA (Dynamic Tensor Analalysis) but with 
    1) Dedicated matrices U per sample (DTA computes U based on a sequence of tensors)
    2) Different input tensor (DTA is adjacency tensor, here is edge-time series for two time resolutions)

    As anomaly score, we measure the energy projected onto the abnormal component (same as in DTA): 
    ||A - hat{A}||_F^2 / ||A||_F^2
    """

    adj_tensor = adj_tensor.coalesce()
    idx = adj_tensor.indices()
    vals = adj_tensor.values()

    # ||X||_F^2 
    adj_tensor_norm2 = (vals * vals).sum()

    # Handle degenerate case of zero tensor
    if adj_tensor_norm2 == 0:
        return 0.0

    # We can work with ||Y||_F^2 due to orthogonality of U
    (ev1, U1), (ev2, U2), (ev3, U3), core = hosvd_sparse(adj_tensor, ranks)
    core_norm2 = (core * core).sum()

    err = 1 - (core_norm2/(adj_tensor_norm2 + 1e-8))
    return err.item()


def swtf_adj_tensor(adj_tensor, num_time_folds):
    '''
    Transform the N x N x T adj tensor into the
    swtf tensor of shape N**2 x fold_size x num_time_folds
    '''

    # Get coords
    idx = adj_tensor.indices()
    vals = adj_tensor.values()
    I1, I2, I3 = adj_tensor.shape
    i1, i2, i3 = idx[0], idx[1], idx[2]
    fold_size = math.ceil(I3/num_time_folds)

    # Flatten (i1, i2)
    e = i1 * I1 + i2

    # Fold time
    f = i3 % fold_size
    b = i3 // fold_size

    # New coords
    new_idx = torch.stack([e, f, b], dim=0)
    new_size = (I1 * I2, fold_size, num_time_folds)
    adj_swtf = torch.sparse_coo_tensor(new_idx, vals, size=new_size).coalesce()

    return adj_swtf


#####################################################
# Main program
#####################################################
def main():

    # Load config
    config = get_config()
 
    # Params
    test_dir = config['paths']['test_dir']
    score_file_path = config['paths']['score_file']

    num_time_folds = int(config['model']['num_time_folds'])
    rank_mode_1 = int(config['model']['rank_mode_1'])
    rank_mode_2 = int(config['model']['rank_mode_2'])
    rank_mode_3 = int(config['model']['rank_mode_3'])

    # Dirs
    set_dir(os.path.dirname(score_file_path))

    # Dataset
    dataset = WindowDataset(test_dir)

    # Load
    test_dataset = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_single,
    )

    # Verify sparse tensors
    torch.sparse.check_sparse_tensor_invariants.enable()

    # Anomaly scoring
    anomaly_scores = []
    for ix, sample in enumerate(test_dataset):

        #print(f'Processing sample {ix}')

        adj_swtf = swtf_adj_tensor(sample['A'], num_time_folds)
        score = swtf_scoring(adj_swtf, ranks=(rank_mode_1, rank_mode_2, rank_mode_3))
        anomaly_scores.append( score )

    
    with open(score_file_path, "wb") as f:
        pickle.dump(anomaly_scores, f)


if __name__ == "__main__":
    main()
