# SWTF

This repository contains an **independent Python implementation** of the SWTF anomaly detection method proposed by Ming Xu, Jianping Wu, Haohan Wang, and Mengxin Cao in [*Anomaly Detection in Road Networks Using Sliding-Window Tensor Factorization* (IEEE Transactions on Intelligent Transportation Systems, 2019)](https://doi.org/10.1109/TITS.2019.2941649). It was developed as a baseline for the experimental evaluation of **MuTGN** and is **not** the original authors' software.

The implementation detects **window-level anomalies** in temporal graphs. Each window corresponds to interactions occurring within a contiguous time interval and is represented by a sparse adjacency tensor. The code expects preprocessed windows in chronological order and assigns one continuous anomaly score to each window. 

The present implementation uses the HOSVD-based formulation of SWTF and produces a single anomaly score by combining reconstruction errors across tensor modes.

## Method

The method operates on temporal graph windows reshaped to represent edge activity at fine-grained and coarse-grained time scales. Each window is initially represented by a sparse adjacency tensor $A \in \mathbb{R}^{N \times N \times T}$, where $N$ is the number of nodes and $T$ is the number of time steps. The two node modes are then flattened into a single dimension representing $N^2$ possible node pairs, while the temporal dimension is divided into $F$ consecutive folds, each containing up to $L$ time steps. The resulting third-order tensor is

$$ X \in \mathbb{R}^{N^2 \times L \times F}, \qquad L = \left\lceil \frac{T}{F} \right\rceil. $$

The original SWTF paper detects mode-wise anomalies by applying thresholding rules to reconstruction errors obtained from $\hat{X}^{(m)} = G^{(m)} \times_m U_m$, where $G^{(m)} = X \times_m U_m^\top$ is the projection of $X$ along mode $m$, and $U_m$ contains the leading left singular vectors of the corresponding mode-wise matricization of $X$.

The present implementation instead projects $X$ along all three modes and computes a single continuous anomaly score from the resulting joint reconstruction error:

$$ score(X) =\frac{\|X-\hat X\|_F^2}{\|X\|_F^2} = 1 - \frac{\|G\|_F^2}{\|X\|_F^2}, $$

where $G = X \times_1 U_1^\top \times_2 U_2^\top \times_3 U_3^\top$ and $\hat X = G \times_1 U_1 \times_2 U_2 \times_3 U_3$. The second equality follows from the orthonormality of the columns of $U_m$.

The implementation adds $10^{-8}$ to the denominator of the final expression for numerical stability. An empty (zero-norm) window is assigned a score of `0.0`.

## Implementation choices relative to the original paper

This implementation focuses on producing one window-level anomaly score for the MuTGN experimental evaluation. It does not reproduce every component or application described in the original paper. In particular:

- **Projection computation:** the present implementation computes the matrices $U_m$ from scratch for each window using HOSVD, whereas the original SWTF paper proposes Bi-SVD (Algorithm 1) to efficiently update estimates from the previous window. Computing these matrices independently avoids propagating approximation errors between windows, albeit at the cost of not exploiting the acceleration. 
- **Anomaly scoring:** the original paper uses mode-wise reconstruction errors and thresholding rules to identify different anomaly types. This implementation instead computes a single tensor-level score based on the joint normalized squared reconstruction error. This corresponds to the square of the relative reconstruction error defined in Equation (5) of the paper.
- **Rank selection:** the implementation uses separately configurable ranks for each mode, whereas the original paper proposes selecting ranks from the knee of the mode-wise reconstruction-error curves.
- **Scope:** the implementation does not include anomaly-type classification, binary anomaly decisions, or path-level anomaly inference.

## Configuration

The main entry point is `test.py`, controlled through a YAML file passed with `--config`. An example is provided in [`config.example.yaml`](config.example.yaml):

```yaml
model:
  num_time_folds: 1
  rank_mode_1: 8
  rank_mode_2: 4
  rank_mode_3: 1

paths:
  test_dir: /data/test
  score_file: /outputs/scores.pkl
```

`model.num_time_folds` defines the number of folds and `rank_mode_1`, `rank_mode_2`, and `rank_mode_3` specify the requested rank for each mode. 

All paths in this example refer to locations **inside the container**.

## Input and output

The implementation expects a directory of preprocessed windows, specified by `paths.test_dir`. Files must match the name pattern `window_*.pt` and are loaded in **lexicographic filename order**. Filenames must therefore preserve chronological order, for example by using zero-padded indices (`window_0000.pt`, `window_0001.pt`, and so on).

Each file is read with `torch.load()` and must contain the window's adjacency tensor under the key `A`, represented as a **three-dimensional sparse PyTorch COO tensor** of shape $N\times N\times T$. The code reads its nonzero indices, values, and shape; the input should be coalesced so these coordinates can be accessed. The first two dimensions correspond to source and destination node indices, and the third to time steps within the window. $N$ and $T$ should remain stable across windows. 

The detection script (`test.py`) writes a pickle file at `paths.score_file`. It contains a **Python list of floating-point anomaly scores** in the same order as the input windows. Window preparation and evaluation against ground-truth labels are handled outside this repository. 

## Reproducible environment

A reconstructed environment using **Python 3.15.2** and the versions pinned in [`requirements-reconstructed.txt`](requirements-reconstructed.txt) has been validated in a local virtual environment and a container. The main dependencies are PyTorch `2.15.0`, NumPy `2.5.5`, SciPy `1.18.1`, and PyYAML `6.0.4`. 

The [`Dockerfile`](Dockerfile) uses `python:3.15.2-slim-trixie`. From the repository root, build the image with [Podman](https://podman.io/) (Docker can also be used):

```bash
podman build --no-cache -t swtf:locked .
```

Verify that the principal dependencies import successfully:

```bash
podman run --rm swtf:locked \
  python -c "import torch, numpy, scipy, yaml; print('SWTF imports OK')"
```

The image runs `test.py` on demand. The input windows, configuration, and anomaly-score output are kept outside the image.

## Running window-level anomaly detection

Prepare a host directory containing the preprocessed test windows. From the repository root, copy and adjust the example configuration, then run the container:

```bash
cp config.example.yaml config.yaml
mkdir -p outputs

TEST_WINDOWS="/absolute/path/to/preprocessed/windows/test"

podman run --rm \
  -v "$PWD/config.yaml:/test_config.yaml:ro,z" \
  -v "$TEST_WINDOWS:/data/test:ro,z" \
  -v "$PWD/outputs:/outputs:Z" \
  swtf:locked \
  python -u /app/test.py --config /test_config.yaml
```

On systems with SELinux enabled, `:z` and `:Z` apply the appropriate labels to volume mounts. These options can be omitted on systems where SELinux is disabled.

## Repository contents

| File | Purpose |
| --- | --- |
| `test.py` | Main window-level factorization and anomaly-score export |
| `data.py` | Discover and load preprocessed sparse `.pt` window samples |
| `config.py` | Read the YAML configuration and create output directories |
| `Dockerfile` | Reconstructed container environment |
| `requirements-reconstructed.txt` | Pinned dependency versions |
| `config.example.yaml` | Example inference configuration used for a validated experiment |

## License and attribution

This repository contains an independent implementation developed for the MuTGN benchmark. It is not the official software of Xu, Wu, Wang, and Cao.

The code is released under the **European Union Public Licence, version 1.2 (EUPL-1.2)**. The license text is provided in [`LICENSE.txt`](LICENSE.txt).

## Citation

If you use the SWTF method, please cite the original paper:

```bibtex
@article{xu2019anomaly,
  title   = {Anomaly Detection in Road Networks Using Sliding-Window Tensor Factorization},
  author  = {Xu, Ming and Wu, Jianping and Wang, Haohan and Cao, Mengxin},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume  = {20},
  number  = {12},
  pages   = {4704--4713},
  year    = {2019},
  doi     = {10.1109/TITS.2019.2941649}
}
```

For the original SWTF algorithm, mode-specific anomaly detection, and road-network experiments, see the [IEEE paper](https://doi.org/10.1109/TITS.2019.2941649).
