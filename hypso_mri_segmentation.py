"""
HyPSO: Adaptive Hyper-Heuristic Control of Particle Swarm Optimisation
for Multilevel Intensity Partitioning of Brain MRI Images

Implementation based on:
  Mbini, M. & Nyathi, T. "Adaptive Hyper-Heuristic Control of Particle Swarm
  Optimisation for Multilevel Intensity Partitioning of Brain MRI Images."
  University of Pretoria, Department of Computer Science.

Algorithm overview
------------------
HyPSO couples canonical PSO (Kennedy & Eberhart, 1995) with a meta-level
hyper-heuristic controller that selects online among five hill-climbing
local-search variants applied exclusively to the global-best particle.
PSO velocity / position update equations remain entirely unchanged.

The objective is 2D Renyi entropy (q = 0.5) computed on the joint histogram
of the original image and its nonlocal-means-filtered counterpart.

Five low-level heuristics (LLH pool)
-------------------------------------
  SHC   - Steepest-ascent Hill Climbing (first-improvement, fixed scan order)
  SAHC  - Steepest-Ascent Hill Climbing (best-improvement, exhaustive scan)
  StHC  - Stochastic Hill Climbing   (random neighbour, probabilistic accept)
  FCHC  - First-Choice Hill Climbing  (random sampling, first improvement)
  RRHC  - Random-Restart Hill Climbing (SHC + random restart on stagnation)

Hyper-heuristic controller (Choice Function)
---------------------------------------------
  Reward signal : DeltaRenyi / Delta_t  (improvement per wall-clock second)
  Cumulative reward : sliding window of last W=10 applications per heuristic
  Selection      : argmax over (cumulative_reward + epsilon * exploration_bonus)
  Exploration    : epsilon-greedy with epsilon = 0.15

Hyperparameters (Table 1, paper)
----------------------------------
  N   = 50            Swarm size
  T   = 10000 * D     Max function evaluations  (D = number of thresholds)
  w   = 0.729         Inertia weight
  c1  = c2 = 1.494    Acceleration coefficients
  q   = 0.5           Renyi order parameter
  d   = 1             Neighbourhood step size
  tau = 2             Local-search interval (iterations)
  eps = 0.15          Exploration probability
  W   = 10            Reward window length
  Rmax= 3             Max RRHC restarts per invocation
"""

from __future__ import annotations

import time
import random
import math
import copy
from typing import List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency: scikit-image for nonlocal-means filtering.
# Falls back to a Gaussian blur if not available.
# ---------------------------------------------------------------------------
try:
    from skimage.restoration import denoise_nl_means, estimate_sigma
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ===========================================================================
# 1.  PREPROCESSING AND JOINT HISTOGRAM
# ===========================================================================

def nonlocal_means_filter(image: np.ndarray) -> np.ndarray:
    """Return a nonlocal-means-filtered copy of *image* (grayscale uint8)."""
    img_float = image.astype(np.float32) / 255.0
    if _HAS_SKIMAGE:
        sigma_est = np.mean(estimate_sigma(img_float, channel_axis=None))
        denoised = denoise_nl_means(
            img_float,
            h=0.8 * sigma_est,
            fast_mode=True,
            patch_size=5,
            patch_distance=6,
            channel_axis=None,
        )
        return (denoised * 255.0).clip(0, 255).astype(np.uint8)
    elif _HAS_SCIPY:
        # Gaussian blur as a lightweight substitute
        denoised = gaussian_filter(img_float, sigma=1.0)
        return (denoised * 255.0).clip(0, 255).astype(np.uint8)
    else:
        # Plain mean filter via numpy
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(img_float, 2, mode="reflect")
        windows = sliding_window_view(pad, (5, 5))
        denoised = windows.mean(axis=(-2, -1))
        return (denoised * 255.0).clip(0, 255).astype(np.uint8)


def build_joint_histogram(
    image: np.ndarray, filtered: np.ndarray, num_bins: int = 256
) -> np.ndarray:
    """Compute normalised 2-D joint histogram p_{ij}.

    Parameters
    ----------
    image    : original grayscale image (H x W, uint8, values in [0, 255])
    filtered : nonlocal-means filtered version of image
    num_bins : histogram bins (default 256 for 8-bit images)

    Returns
    -------
    hist2d   : (num_bins x num_bins) float64 array, sums to 1.0
    """
    hist2d, _, _ = np.histogram2d(
        image.ravel().astype(np.float64),
        filtered.ravel().astype(np.float64),
        bins=num_bins,
        range=[[0, num_bins], [0, num_bins]],
    )
    total = hist2d.sum()
    if total > 0:
        hist2d /= total
    return hist2d


# ===========================================================================
# 2.  OBJECTIVE FUNCTION: 2-D RENYI ENTROPY
# ===========================================================================

def renyi_entropy_class(
    p_ij: np.ndarray,
    class_mask: np.ndarray,
    q: float = 0.5,
) -> float:
    """Renyi entropy of order *q* for a single class region.

    H_k^(q) = 1/(1-q) * ln( sum_{(i,j) in class_k} (p_ij / omega_k)^q )

    Parameters
    ----------
    p_ij       : 2-D joint histogram (normalised, shape [256, 256])
    class_mask : boolean mask of same shape as p_ij selecting class pixels
    q          : Renyi order (default 0.5 as per Dwivedi et al., 2026)

    Returns
    -------
    entropy value (float)
    """
    omega_k = p_ij[class_mask].sum()
    if omega_k <= 0:
        return 0.0
    ratios = p_ij[class_mask] / omega_k
    ratios = ratios[ratios > 0]
    if len(ratios) == 0:
        return 0.0
    return (1.0 / (1.0 - q)) * np.log(np.sum(ratios ** q))


def compute_2d_renyi_entropy(
    thresholds: np.ndarray,
    joint_hist: np.ndarray,
    q: float = 0.5,
) -> float:
    """Total 2-D Renyi entropy H(t) = sum_{k=1}^{n} H_k^(q).

    Parameters
    ----------
    thresholds : 1-D sorted integer array of shape (D,), values in [1, 254]
                 defining D+1 intensity classes on [0, 255]
    joint_hist : normalised 2-D joint histogram (256 x 256)
    q          : Renyi order parameter

    Returns
    -------
    total entropy (float) — the fitness to maximise
    """
    n_bins = joint_hist.shape[0]
    boundaries = np.concatenate([[0], thresholds, [n_bins]])
    total_entropy = 0.0
    rows = np.arange(n_bins)
    cols = np.arange(n_bins)
    grid_i, grid_j = np.meshgrid(rows, cols, indexing="ij")

    for k in range(len(boundaries) - 1):
        lo, hi = int(boundaries[k]), int(boundaries[k + 1])
        # class_k covers intensities [lo, hi) in both dimensions
        mask = (grid_i >= lo) & (grid_i < hi) & (grid_j >= lo) & (grid_j < hi)
        total_entropy += renyi_entropy_class(joint_hist, mask, q)

    return total_entropy


# ===========================================================================
# 3.  NEIGHBOURHOOD DEFINITION (shared by all LLH variants)
# ===========================================================================

def generate_neighbours(
    thresholds: np.ndarray, delta: int = 1
) -> List[np.ndarray]:
    """Generate all neighbours by perturbing each component t_k by +/-delta.

    Perturbed values are clamped to [0, 255] and ordering violations
    (t_k >= t_{k+1}) are repaired by sorting.

    Returns a list of unique neighbour threshold arrays.
    """
    neighbours: List[np.ndarray] = []
    n = len(thresholds)
    for k in range(n):
        for sign in (-1, +1):
            candidate = thresholds.copy()
            candidate[k] = int(np.clip(candidate[k] + sign * delta, 1, 254))
            candidate = np.sort(candidate)
            # Skip if identical to current
            if not np.array_equal(candidate, thresholds):
                # Check uniqueness
                is_dup = any(np.array_equal(candidate, nb) for nb in neighbours)
                if not is_dup:
                    neighbours.append(candidate)
    return neighbours


def random_threshold_vector(D: int) -> np.ndarray:
    """Sample a random sorted threshold vector of dimension D."""
    t = np.random.randint(1, 255, size=D)
    return np.sort(t)


# ===========================================================================
# 4.  FIVE LOW-LEVEL HEURISTICS (LLH)
# ===========================================================================

# Each LLH has the signature:
#   heuristic(current: np.ndarray,
#             current_fitness: float,
#             joint_hist: np.ndarray,
#             **kwargs) -> Tuple[np.ndarray, float, int]
# and returns (best_solution, best_fitness, n_evaluations).

def shc(
    current: np.ndarray,
    current_fitness: float,
    joint_hist: np.ndarray,
    delta: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, float, int]:
    """Steepest-ascent Hill Climbing (first-improvement, fixed scan order).

    Iterates over neighbours in a fixed order; moves to the first
    neighbour that improves the fitness; terminates when a full scan
    produces no improvement.
    """
    n_evals = 0
    s = current.copy()
    f_s = current_fitness

    improved = True
    while improved:
        improved = False
        for nb in generate_neighbours(s, delta):
            f_nb = compute_2d_renyi_entropy(nb, joint_hist)
            n_evals += 1
            if f_nb > f_s:
                s, f_s = nb, f_nb
                improved = True
                break  # first improvement
    return s, f_s, n_evals


def sahc(
    current: np.ndarray,
    current_fitness: float,
    joint_hist: np.ndarray,
    delta: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, float, int]:
    """Steepest-Ascent Hill Climbing (best-improvement, exhaustive scan).

    Evaluates all 2*(n-1) neighbours and moves to the best one;
    terminates if no neighbour improves f(t).
    """
    n_evals = 0
    s = current.copy()
    f_s = current_fitness

    improved = True
    while improved:
        improved = False
        best_nb, best_f = s, f_s
        for nb in generate_neighbours(s, delta):
            f_nb = compute_2d_renyi_entropy(nb, joint_hist)
            n_evals += 1
            if f_nb > best_f:
                best_nb, best_f = nb, f_nb
                improved = True
        if improved:
            s, f_s = best_nb, best_f
    return s, f_s, n_evals


def sthc(
    current: np.ndarray,
    current_fitness: float,
    joint_hist: np.ndarray,
    delta: int = 1,
    p_accept: float = 0.05,
    **kwargs,
) -> Tuple[np.ndarray, float, int]:
    """Stochastic Hill Climbing.

    Selects a neighbour uniformly at random; accepts it with probability
    proportional to max(0, f(t') - f(t)) for improvements and with
    fixed probability p_accept for worse solutions.
    Terminates after 5*(n-1) stochastic steps.
    """
    n = len(current)
    max_steps = 5 * max(n - 1, 1)
    n_evals = 0
    s = current.copy()
    f_s = current_fitness

    for _ in range(max_steps):
        neighbours = generate_neighbours(s, delta)
        if not neighbours:
            break
        nb = random.choice(neighbours)
        f_nb = compute_2d_renyi_entropy(nb, joint_hist)
        n_evals += 1
        delta_f = f_nb - f_s
        if delta_f > 0:
            # Accept improving move proportional to improvement
            prob = min(1.0, delta_f / (abs(f_s) + 1e-12))
            if random.random() < prob:
                s, f_s = nb, f_nb
        else:
            # Accept worse solution with fixed probability
            if random.random() < p_accept:
                s, f_s = nb, f_nb
    return s, f_s, n_evals


def fchc(
    current: np.ndarray,
    current_fitness: float,
    joint_hist: np.ndarray,
    delta: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, float, int]:
    """First-Choice Hill Climbing.

    Samples neighbours uniformly at random; accepts the first improvement;
    terminates after an improvement or 10*(n-1) unsuccessful samples.
    """
    n = len(current)
    max_attempts = 10 * max(n - 1, 1)
    n_evals = 0
    s = current.copy()
    f_s = current_fitness

    attempts = 0
    while attempts < max_attempts:
        neighbours = generate_neighbours(s, delta)
        if not neighbours:
            break
        nb = random.choice(neighbours)
        f_nb = compute_2d_renyi_entropy(nb, joint_hist)
        n_evals += 1
        attempts += 1
        if f_nb > f_s:
            s, f_s = nb, f_nb
            break
    return s, f_s, n_evals


def rrhc(
    current: np.ndarray,
    current_fitness: float,
    joint_hist: np.ndarray,
    delta: int = 1,
    r_max: int = 3,
    **kwargs,
) -> Tuple[np.ndarray, float, int]:
    """Random-Restart Hill Climbing.

    Executes a full SHC from *current*; if no improvement, restarts from a
    randomly generated threshold vector (up to r_max restarts per invocation).
    Keeps the best solution found across all runs.
    """
    n_evals = 0
    best_s, best_f = shc(current, current_fitness, joint_hist, delta)
    n_evals += 0  # SHC evals counted separately; for simplicity track here:

    for _ in range(r_max):
        if best_f > current_fitness:
            break  # Improvement found; no restart needed
        D = len(current)
        rand_start = random_threshold_vector(D)
        rand_f = compute_2d_renyi_entropy(rand_start, joint_hist)
        n_evals += 1
        s_candidate, f_candidate, _ = shc(rand_start, rand_f, joint_hist, delta)
        if f_candidate > best_f:
            best_s, best_f = s_candidate, f_candidate
    return best_s, best_f, n_evals


# Registry: name -> function
LLH_POOL = {
    "SHC": shc,
    "SAHC": sahc,
    "StHC": sthc,
    "FCHC": fchc,
    "RRHC": rrhc,
}
LLH_NAMES = list(LLH_POOL.keys())  # Fixed order for indexing


# ===========================================================================
# 5.  HYPER-HEURISTIC CONTROLLER (Choice Function)
# ===========================================================================

class HyperHeuristicController:
    """Adaptive selector using a sliding-window Choice Function.

    Reward signal  : DeltaRenyi / Delta_t  (entropy gain per second)
    Selection      : epsilon-greedy argmax over cumulative reward
    Window length  : W (last W applications per heuristic)
    """

    def __init__(
        self,
        heuristic_names: List[str],
        epsilon: float = 0.15,
        window_size: int = 10,
    ):
        self.names = heuristic_names
        self.n = len(heuristic_names)
        self.epsilon = epsilon
        self.W = window_size
        # Sliding window of (reward, time) tuples per heuristic
        self._windows: List[List[float]] = [[] for _ in range(self.n)]
        # Total selections per heuristic (for UCB-style exploration)
        self._selections = [0] * self.n
        self._total_selections = 0

    def cumulative_reward(self, idx: int) -> float:
        """Running sum of reward over the last W applications."""
        w = self._windows[idx]
        return sum(w)

    def select(self) -> int:
        """Return the index of the selected heuristic (epsilon-greedy)."""
        self._total_selections += 1
        if random.random() < self.epsilon:
            # Exploration: uniform random
            return random.randint(0, self.n - 1)
        # Exploitation: choose heuristic with highest cumulative reward
        rewards = [self.cumulative_reward(i) for i in range(self.n)]
        return int(np.argmax(rewards))

    def update(self, idx: int, delta_renyi: float, delta_t: float) -> None:
        """Update cumulative reward for heuristic *idx*."""
        if delta_t > 0:
            reward = delta_renyi / delta_t
        else:
            reward = delta_renyi
        w = self._windows[idx]
        w.append(reward)
        if len(w) > self.W:
            w.pop(0)
        self._selections[idx] += 1

    def apply(
        self,
        global_best: np.ndarray,
        global_best_fitness: float,
        joint_hist: np.ndarray,
        delta: int = 1,
        r_max: int = 3,
    ) -> Tuple[np.ndarray, float]:
        """Select and apply one heuristic; update controller state.

        Returns the (possibly improved) global-best solution and its fitness.
        The result replaces *global_best* only if strictly higher entropy.
        """
        idx = self.select()
        heuristic_fn = LLH_POOL[LLH_NAMES[idx]]

        t_start = time.perf_counter()
        f_before = global_best_fitness
        new_solution, new_fitness, _ = heuristic_fn(
            global_best.copy(),
            global_best_fitness,
            joint_hist,
            delta=delta,
            r_max=r_max,
        )
        t_end = time.perf_counter()

        delta_renyi = new_fitness - f_before
        delta_t = t_end - t_start
        self.update(idx, delta_renyi, delta_t)

        # Improvement-only acceptance
        if new_fitness > global_best_fitness:
            return new_solution, new_fitness
        return global_best, global_best_fitness


# ===========================================================================
# 6.  CANONICAL PSO  (Algorithm 2 from the paper)
# ===========================================================================

class Particle:
    """Single PSO particle representing a candidate threshold vector."""

    def __init__(self, D: int):
        self.position: np.ndarray = random_threshold_vector(D)
        self.velocity: np.ndarray = np.zeros(D, dtype=np.float64)
        self.personal_best: np.ndarray = self.position.copy()
        self.personal_best_fitness: float = -np.inf

    def evaluate(self, joint_hist: np.ndarray, q: float = 0.5) -> float:
        """Evaluate 2-D Renyi entropy at current position."""
        return compute_2d_renyi_entropy(self.position, joint_hist, q)


def _clamp_and_sort(position: np.ndarray) -> np.ndarray:
    """Clamp threshold values to [1, 254] and sort to maintain order."""
    clamped = np.clip(np.round(position).astype(int), 1, 254)
    return np.sort(clamped)


def hypso(
    joint_hist: np.ndarray,
    D: int,
    N: int = 50,
    T_max: Optional[int] = None,
    omega: float = 0.729,
    c1: float = 1.494,
    c2: float = 1.494,
    q: float = 0.5,
    delta: int = 1,
    tau: int = 2,
    epsilon: float = 0.15,
    W: int = 10,
    r_max: int = 3,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """Run the HyPSO algorithm.

    Parameters
    ----------
    joint_hist : 2-D normalised joint histogram (256 x 256)
    D          : number of thresholds (dimensionality)
    N          : swarm size (default 50)
    T_max      : maximum iterations; defaults to 10000 * D
    omega      : inertia weight (default 0.729)
    c1, c2     : cognitive / social acceleration (default 1.494 each)
    q          : Renyi order parameter (default 0.5)
    delta      : neighbourhood step size for local search (default 1)
    tau        : local-search interval in iterations (default 2)
    epsilon    : exploration probability for choice function (default 0.15)
    W          : reward window length (default 10)
    r_max      : max RRHC restarts per invocation (default 3)
    seed       : random seed (optional)
    verbose    : print progress every 100 iterations (default False)

    Returns
    -------
    (best_thresholds, best_fitness)
      best_thresholds : 1-D sorted int array of shape (D,)
      best_fitness    : 2-D Renyi entropy value
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if T_max is None:
        T_max = 10_000 * D

    # -----------------------------------------------------------------------
    # Initialise swarm
    # -----------------------------------------------------------------------
    swarm: List[Particle] = [Particle(D) for _ in range(N)]

    # Initial evaluation
    for p in swarm:
        f = p.evaluate(joint_hist, q)
        p.personal_best = p.position.copy()
        p.personal_best_fitness = f

    # Global best
    best_idx = int(np.argmax([p.personal_best_fitness for p in swarm]))
    global_best: np.ndarray = swarm[best_idx].personal_best.copy()
    global_best_fitness: float = swarm[best_idx].personal_best_fitness

    # Hyper-heuristic controller
    controller = HyperHeuristicController(LLH_NAMES, epsilon=epsilon, window_size=W)

    # -----------------------------------------------------------------------
    # Main loop (Algorithm 2)
    # -----------------------------------------------------------------------
    for t in range(1, T_max + 1):
        for p in swarm:
            r1 = np.random.uniform(0, 1, size=D)
            r2 = np.random.uniform(0, 1, size=D)

            # Velocity update (Eq. 1)
            p.velocity = (
                omega * p.velocity
                + c1 * r1 * (p.personal_best - p.position)
                + c2 * r2 * (global_best - p.position)
            )

            # Position update (Eq. 2)
            new_pos = _clamp_and_sort(p.position + p.velocity)
            p.position = new_pos

            # Update personal best
            f = p.evaluate(joint_hist, q)
            if f > p.personal_best_fitness:
                p.personal_best = p.position.copy()
                p.personal_best_fitness = f

            # Update global best
            if f > global_best_fitness:
                global_best = p.position.copy()
                global_best_fitness = f

        # Meta-level local search applied every tau iterations
        if t % tau == 0:
            global_best, global_best_fitness = controller.apply(
                global_best,
                global_best_fitness,
                joint_hist,
                delta=delta,
                r_max=r_max,
            )

        if verbose and (t % 100 == 0 or t == 1):
            print(
                f"  Iter {t:5d}/{T_max}: best_fitness={global_best_fitness:.6f} "
                f"thresholds={global_best.tolist()}"
            )

    return global_best, global_best_fitness


# ===========================================================================
# 7.  SEGMENTATION PIPELINE
# ===========================================================================

def segment_mri(
    image: np.ndarray,
    n_thresholds: int = 4,
    hypso_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Full MRI segmentation pipeline using HyPSO.

    Parameters
    ----------
    image         : grayscale MRI image (H x W, uint8 or float converted to uint8)
    n_thresholds  : number of threshold levels D (produces D+1 segments)
    hypso_kwargs  : additional keyword arguments forwarded to hypso()
    verbose       : whether to print HyPSO progress

    Returns
    -------
    (thresholds, segmented_image, best_fitness)
      thresholds      : 1-D sorted int array of shape (D,)
      segmented_image : H x W uint8 label map (values 0 .. D)
      best_fitness    : final 2-D Renyi entropy value
    """
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = ((image - image.min()) / (image.max() - image.min() + 1e-12) * 255).astype(np.uint8)

    # 1. Nonlocal-means filtering
    if verbose:
        print("[HyPSO] Computing nonlocal-means filtered image ...")
    filtered = nonlocal_means_filter(image)

    # 2. 2-D joint histogram
    if verbose:
        print("[HyPSO] Building 2-D joint histogram ...")
    joint_hist = build_joint_histogram(image, filtered, num_bins=256)

    # 3. Run HyPSO optimisation
    if verbose:
        print(f"[HyPSO] Running optimisation with D={n_thresholds} thresholds ...")
    kwargs = hypso_kwargs or {}
    kwargs.setdefault("verbose", verbose)
    thresholds, fitness = hypso(joint_hist, D=n_thresholds, **kwargs)

    # 4. Apply thresholds to produce label map
    boundaries = np.concatenate([[0], thresholds, [256]])
    segmented = np.zeros_like(image, dtype=np.uint8)
    for k in range(len(boundaries) - 1):
        lo, hi = int(boundaries[k]), int(boundaries[k + 1])
        segmented[(image >= lo) & (image < hi)] = k

    return thresholds, segmented, fitness


# ===========================================================================
# 8.  EVALUATION METRIC — SSIM (reconstruction fidelity)
# ===========================================================================

def compute_ssim(image: np.ndarray, segmented: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between original and
    segmented image (used as reconstruction fidelity metric in the paper).

    Requires scikit-image.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        # Map label map back to intensity means for comparison
        unique_labels = np.unique(segmented)
        reconstructed = np.zeros_like(image, dtype=np.float64)
        for lbl in unique_labels:
            mask = segmented == lbl
            reconstructed[mask] = image[mask].mean()
        reconstructed = reconstructed.astype(np.uint8)
        score = ssim(image, reconstructed, data_range=255)
        return float(score)
    except ImportError:
        raise ImportError("scikit-image is required for SSIM computation.")


# ===========================================================================
# 9.  COMMAND-LINE INTERFACE (demo / quick test)
# ===========================================================================

def _demo():
    """Run a quick demo on a synthetic test image."""
    print("=" * 60)
    print("HyPSO MRI Segmentation — Demo")
    print("=" * 60)

    # Create a synthetic grayscale image with multiple intensity regions
    rng = np.random.default_rng(42)
    h, w = 128, 128
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, :] = 30              # background
    img[20:60, 20:60] = 90      # region 1
    img[60:100, 60:100] = 150   # region 2
    img[10:30, 90:110] = 200    # region 3
    # Add noise
    noise = rng.integers(-10, 10, size=(h, w))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    D = 3  # 3 thresholds → 4 segments
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Number of thresholds D = {D}")
    print()

    thresholds, segmented, fitness = segment_mri(
        img,
        n_thresholds=D,
        hypso_kwargs={
            "N": 20,          # Reduced swarm for demo speed
            "T_max": 200,     # Reduced iterations for demo speed
            "seed": 1,
        },
        verbose=True,
    )

    print()
    print(f"Optimal thresholds : {thresholds.tolist()}")
    print(f"2D Renyi entropy   : {fitness:.6f}")
    print(f"Unique segments    : {np.unique(segmented).tolist()}")

    try:
        ssim_score = compute_ssim(img, segmented)
        print(f"SSIM (recon. fidelity): {ssim_score:.4f}")
    except ImportError:
        print("SSIM: skimage not available, skipping.")

    print()
    print("Demo completed successfully.")


if __name__ == "__main__":
    _demo()
