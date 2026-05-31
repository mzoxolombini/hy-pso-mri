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
  Reward signal  : DeltaRenyi / Delta_t  (improvement per wall-clock second)
  Cumulative reward : sliding window of last W=10 applications per heuristic
  Selection      : epsilon-greedy argmax over cumulative reward
  Exploration    : epsilon-greedy with epsilon = 0.15

Hyperparameters (Table 1, paper)
----------------------------------
  N     = 50           Swarm size
  FE_max= 10,000 * D   Max function evaluations  (D = number of thresholds)
  w     = 0.729        Inertia weight
  c1=c2 = 1.494        Acceleration coefficients
  q     = 0.5          Renyi order parameter
  delta = 1            Neighbourhood step size
  tau   = 2            Local-search interval (iterations)
  eps   = 0.15         Exploration probability
  W     = 10           Reward window length
  Rmax  = 3            Max RRHC restarts per invocation

Fixes applied vs. initial version
-----------------------------------
  [FIX-1] T_max is now correctly treated as a function-evaluation budget
          (FE_max = 10,000 * D).  Each PSO iteration costs N evaluations;
          local-search evaluations are also counted.  The loop exits as soon
          as the budget is exhausted.
  [FIX-2] RRHC restart logic was inverted: the early-exit check was placed
          BEFORE any restart attempt, so restarts never fired.  Fixed by
          checking improvement AFTER each SHC restart run.
  [FIX-3] 2D Renyi entropy class mask now uses the correct marginal row band
          [lo, hi) on the original-image axis (rows) across ALL columns of
          the joint histogram, matching the standard 2D-Renyi MLT formulation.
          The previous square-region mask discarded all off-diagonal mass.
  [FIX-4] Joint histogram meshgrid pre-computed once and cached at module
          level; eliminated per-call rebuild inside the hot fitness function.
  [FIX-5] Neighbour deduplication uses a set of tuples instead of O(n^2)
          array comparisons.
  [FIX-6] StHC acceptance probability fixed: improvements are now always
          accepted (prob = 1.0) and worse solutions accepted with the fixed
          probability p_accept = 0.05, matching the paper description.
  [FIX-7] Input validation added to segment_mri() and hypso().
"""

from __future__ import annotations

import time
import random
from typing import List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
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

# ---------------------------------------------------------------------------
# [FIX-4] Pre-compute the 256x256 row-index grid once at module load time.
# Used by compute_2d_renyi_entropy() on every fitness call.
# ---------------------------------------------------------------------------
_NBINS = 256
_ROW_IDX = np.arange(_NBINS)   # shape (256,) — row indices of joint histogram


# ===========================================================================
# 1.  PREPROCESSING AND JOINT HISTOGRAM
# ===========================================================================

def nonlocal_means_filter(image: np.ndarray) -> np.ndarray:
    """Return a nonlocal-means-filtered copy of *image* (grayscale uint8).

    Uses scikit-image if available; falls back to scipy Gaussian blur, then
    to a plain numpy mean filter.
    """
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
        denoised = gaussian_filter(img_float, sigma=1.0)
        return (denoised * 255.0).clip(0, 255).astype(np.uint8)
    else:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(img_float, 2, mode="reflect")
        windows = sliding_window_view(pad, (5, 5))
        denoised = windows.mean(axis=(-2, -1))
        return (denoised * 255.0).clip(0, 255).astype(np.uint8)


def build_joint_histogram(
    image: np.ndarray,
    filtered: np.ndarray,
    num_bins: int = 256,
) -> np.ndarray:
    """Compute the normalised 2-D joint histogram p_{ij}.

    Parameters
    ----------
    image    : original grayscale image (H x W, uint8, values in [0, 255])
    filtered : nonlocal-means filtered version of image
    num_bins : histogram bins (default 256 for 8-bit images)

    Returns
    -------
    hist2d : (num_bins x num_bins) float64 array summing to 1.0
             Row axis = original image intensity i.
             Col axis = filtered image intensity j.
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

def renyi_entropy_class(row_probs: np.ndarray, q: float = 0.5) -> float:
    """Renyi entropy of order *q* for a single intensity class.

    H_k^(q) = 1/(1-q) * ln( sum_j (p_{kj} / omega_k)^q )

    Parameters
    ----------
    row_probs : 1-D array — sum of joint histogram rows belonging to class k,
                i.e. joint_hist[lo:hi, :].sum(axis=0)  shape (num_bins,)
    q         : Renyi order (default 0.5)

    Returns
    -------
    entropy value (float)
    """
    omega_k = row_probs.sum()
    if omega_k <= 0.0:
        return 0.0
    ratios = row_probs[row_probs > 0] / omega_k
    return (1.0 / (1.0 - q)) * float(np.log(np.sum(ratios ** q)))


def compute_2d_renyi_entropy(
    thresholds: np.ndarray,
    joint_hist: np.ndarray,
    q: float = 0.5,
) -> float:
    """Total 2-D Renyi entropy  H(t) = sum_{k=1}^{n} H_k^(q).

    [FIX-3] Class k uses the marginal row band [lo, hi) across ALL columns
    of the joint histogram (i.e. joint_hist[lo:hi, :]).  This correctly
    captures every pixel whose original intensity falls in class k,
    regardless of the filtered-image intensity.

    Parameters
    ----------
    thresholds : 1-D sorted int array of shape (D,), values in [1, 254]
    joint_hist : normalised 2-D joint histogram, shape (256, 256)
    q          : Renyi order parameter

    Returns
    -------
    total entropy (float) — the fitness to be maximised
    """
    boundaries = np.concatenate([[0], thresholds, [joint_hist.shape[0]]])
    total_entropy = 0.0
    for k in range(len(boundaries) - 1):
        lo, hi = int(boundaries[k]), int(boundaries[k + 1])
        if lo >= hi:
            continue
        # Sum all columns for rows in [lo, hi): shape (num_bins,)
        row_probs = joint_hist[lo:hi, :].sum(axis=0)
        total_entropy += renyi_entropy_class(row_probs, q)
    return total_entropy


# ===========================================================================
# 3.  NEIGHBOURHOOD DEFINITION  (shared by all LLH variants)
# ===========================================================================

def generate_neighbours(
    thresholds: np.ndarray, delta: int = 1
) -> List[np.ndarray]:
    """Generate all neighbours by perturbing each t_k by +/- delta.

    [FIX-5] Deduplication uses a set of tuples — O(n) instead of O(n^2).

    Perturbed values are clamped to [1, 254]; ordering violations are
    repaired by sorting.  Returns a list of unique neighbour arrays.
    """
    seen = set()
    neighbours: List[np.ndarray] = []
    current_key = tuple(thresholds.tolist())
    for k in range(len(thresholds)):
        for sign in (-1, +1):
            candidate = thresholds.copy()
            candidate[k] = int(np.clip(candidate[k] + sign * delta, 1, 254))
            candidate = np.sort(candidate)
            key = tuple(candidate.tolist())
            if key != current_key and key not in seen:
                seen.add(key)
                neighbours.append(candidate)
    return neighbours


def random_threshold_vector(D: int) -> np.ndarray:
    """Sample a random sorted threshold vector of dimension D."""
    t = np.random.randint(1, 255, size=D)
    return np.sort(t)


# ===========================================================================
# 4.  FIVE LOW-LEVEL HEURISTICS  (LLH)
# ===========================================================================
# Signature: fn(current, current_fitness, joint_hist, **kwargs)
#            -> (best_solution, best_fitness, n_evaluations)

def shc(
    current: np.ndarray,
    current_fitness: float,
    joint_hist: np.ndarray,
    delta: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, float, int]:
    """Steepest-ascent Hill Climbing — first-improvement, fixed scan order.

    Moves to the first neighbour that improves fitness; terminates when a
    full scan produces no improvement.
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
                break
    return s, f_s, n_evals


def sahc(
    current: np.ndarray,
    current_fitness: float,
    joint_hist: np.ndarray,
    delta: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, float, int]:
    """Steepest-Ascent Hill Climbing — best-improvement, exhaustive scan.

    Evaluates all neighbours and moves to the best one; terminates if no
    neighbour improves f(t).
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

    [FIX-6] Improving neighbours are always accepted (prob = 1.0).
    Worse solutions are accepted with fixed probability p_accept = 0.05.
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
        if f_nb > f_s:
            s, f_s = nb, f_nb          # always accept improvements
        elif random.random() < p_accept:
            s, f_s = nb, f_nb          # occasionally accept worse solutions
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
    for _ in range(max_attempts):
        neighbours = generate_neighbours(s, delta)
        if not neighbours:
            break
        nb = random.choice(neighbours)
        f_nb = compute_2d_renyi_entropy(nb, joint_hist)
        n_evals += 1
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

    [FIX-2] Corrected restart logic: runs SHC from *current* first; then
    performs up to r_max random restarts ONLY when the previous SHC run
    found no improvement.  Keeps the best solution found across all runs.
    """
    n_evals = 0
    D = len(current)

    # First run: SHC from the current global-best position
    best_s, best_f, e = shc(current, current_fitness, joint_hist, delta)
    n_evals += e

    # Restart only when no improvement was found in the previous SHC run
    prev_f = current_fitness
    for _ in range(r_max):
        if best_f > prev_f:
            break   # improvement found — no further restart needed
        rand_start = random_threshold_vector(D)
        rand_f = compute_2d_renyi_entropy(rand_start, joint_hist)
        n_evals += 1
        s_candidate, f_candidate, e = shc(rand_start, rand_f, joint_hist, delta)
        n_evals += e
        prev_f = best_f   # update baseline for next restart check
        if f_candidate > best_f:
            best_s, best_f = s_candidate, f_candidate

    return best_s, best_f, n_evals


# Registry
LLH_POOL = {
    "SHC":  shc,
    "SAHC": sahc,
    "StHC": sthc,
    "FCHC": fchc,
    "RRHC": rrhc,
}
LLH_NAMES = list(LLH_POOL.keys())


# ===========================================================================
# 5.  HYPER-HEURISTIC CONTROLLER  (Choice Function)
# ===========================================================================

class HyperHeuristicController:
    """Adaptive heuristic selector using a sliding-window Choice Function.

    Reward signal  : DeltaRenyi / Delta_t  (entropy gain per second)
    Selection      : epsilon-greedy argmax over cumulative sliding-window reward
    Window length  : W (last W applications per heuristic)
    Exploration    : uniform random with probability epsilon
    """

    def __init__(
        self,
        heuristic_names: List[str],
        epsilon: float = 0.15,
        window_size: int = 10,
    ) -> None:
        self.names = heuristic_names
        self.n = len(heuristic_names)
        self.epsilon = epsilon
        self.W = window_size
        self._windows: List[List[float]] = [[] for _ in range(self.n)]
        self._selections: List[int] = [0] * self.n
        self._total_selections: int = 0

    def cumulative_reward(self, idx: int) -> float:
        """Sliding-window sum of rewards for heuristic *idx*."""
        return sum(self._windows[idx])

    def select(self) -> int:
        """Return the index of the selected heuristic (epsilon-greedy)."""
        self._total_selections += 1
        if random.random() < self.epsilon:
            return random.randint(0, self.n - 1)
        rewards = [self.cumulative_reward(i) for i in range(self.n)]
        return int(np.argmax(rewards))

    def update(self, idx: int, delta_renyi: float, delta_t: float) -> None:
        """Append the reward for this application to heuristic *idx*."""
        reward = delta_renyi / delta_t if delta_t > 0 else delta_renyi
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
    ) -> Tuple[np.ndarray, float, int]:
        """Select and apply one heuristic; update the controller.

        Returns (new_global_best, new_fitness, n_evaluations_used).
        Improvement-only acceptance: replaces global_best only on strict gain.
        """
        idx = self.select()
        heuristic_fn = LLH_POOL[LLH_NAMES[idx]]

        t_start = time.perf_counter()
        new_sol, new_fit, n_evals = heuristic_fn(
            global_best.copy(),
            global_best_fitness,
            joint_hist,
            delta=delta,
            r_max=r_max,
        )
        delta_t = time.perf_counter() - t_start
        delta_renyi = new_fit - global_best_fitness
        self.update(idx, delta_renyi, delta_t)

        if new_fit > global_best_fitness:
            return new_sol, new_fit, n_evals
        return global_best, global_best_fitness, n_evals


# ===========================================================================
# 6.  CANONICAL PSO WITH HYPER-HEURISTIC  (Algorithm 2)
# ===========================================================================

class Particle:
    """Single PSO particle representing a candidate threshold vector."""

    def __init__(self, D: int) -> None:
        self.position: np.ndarray = random_threshold_vector(D)
        self.velocity: np.ndarray = np.zeros(D, dtype=np.float64)
        self.personal_best: np.ndarray = self.position.copy()
        self.personal_best_fitness: float = -np.inf

    def evaluate(self, joint_hist: np.ndarray, q: float = 0.5) -> float:
        return compute_2d_renyi_entropy(self.position, joint_hist, q)


def _clamp_and_sort(position: np.ndarray) -> np.ndarray:
    """Clamp threshold values to [1, 254] and sort to maintain order."""
    return np.sort(np.clip(np.round(position).astype(int), 1, 254))


def hypso(
    joint_hist: np.ndarray,
    D: int,
    N: int = 50,
    FE_max: Optional[int] = None,
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

    [FIX-1] FE_max is the total function-evaluation budget (default 10,000*D).
    Both PSO evaluations (N per iteration) and local-search evaluations are
    counted against this budget.  The loop exits as soon as it is exhausted.

    Parameters
    ----------
    joint_hist : 2-D normalised joint histogram (256 x 256)
    D          : number of thresholds (problem dimension)
    N          : swarm size (default 50)
    FE_max     : max function evaluations; defaults to 10,000 * D
    omega      : inertia weight (default 0.729)
    c1, c2     : cognitive / social acceleration (default 1.494 each)
    q          : Renyi order parameter (default 0.5)
    delta      : neighbourhood step size for local search (default 1)
    tau        : local-search interval in iterations (default 2)
    epsilon    : exploration probability for choice function (default 0.15)
    W          : reward window length (default 10)
    r_max      : max RRHC restarts per invocation (default 3)
    seed       : optional random seed for reproducibility
    verbose    : print progress every 500 evaluations (default False)

    Returns
    -------
    (best_thresholds, best_fitness)
    """
    # [FIX-7] Input validation
    if joint_hist.ndim != 2 or joint_hist.shape[0] != joint_hist.shape[1]:
        raise ValueError("joint_hist must be a square 2-D array.")
    if D < 1:
        raise ValueError("D (number of thresholds) must be >= 1.")
    if N < 1:
        raise ValueError("N (swarm size) must be >= 1.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if FE_max is None:
        FE_max = 10_000 * D

    fe_used = 0  # function-evaluation counter

    # ------------------------------------------------------------------
    # Initialise swarm
    # ------------------------------------------------------------------
    swarm: List[Particle] = [Particle(D) for _ in range(N)]
    for p in swarm:
        f = p.evaluate(joint_hist, q)
        fe_used += 1
        p.personal_best = p.position.copy()
        p.personal_best_fitness = f

    best_idx = int(np.argmax([p.personal_best_fitness for p in swarm]))
    global_best: np.ndarray = swarm[best_idx].personal_best.copy()
    global_best_fitness: float = swarm[best_idx].personal_best_fitness

    controller = HyperHeuristicController(LLH_NAMES, epsilon=epsilon, window_size=W)
    last_verbose_fe = 0

    # ------------------------------------------------------------------
    # Main loop — iterate until evaluation budget is exhausted
    # ------------------------------------------------------------------
    t = 0
    while fe_used < FE_max:
        t += 1
        for p in swarm:
            if fe_used >= FE_max:
                break
            r1 = np.random.uniform(0.0, 1.0, size=D)
            r2 = np.random.uniform(0.0, 1.0, size=D)

            # Velocity update (Eq. 1)
            p.velocity = (
                omega * p.velocity
                + c1 * r1 * (p.personal_best - p.position)
                + c2 * r2 * (global_best   - p.position)
            )
            # Position update (Eq. 2)
            p.position = _clamp_and_sort(p.position + p.velocity)

            # Evaluate and update personal / global best
            f = p.evaluate(joint_hist, q)
            fe_used += 1
            if f > p.personal_best_fitness:
                p.personal_best = p.position.copy()
                p.personal_best_fitness = f
            if f > global_best_fitness:
                global_best = p.position.copy()
                global_best_fitness = f

        # Meta-level local search every tau iterations
        if t % tau == 0 and fe_used < FE_max:
            global_best, global_best_fitness, ls_evals = controller.apply(
                global_best, global_best_fitness, joint_hist,
                delta=delta, r_max=r_max,
            )
            fe_used += ls_evals

        if verbose and (fe_used - last_verbose_fe) >= 500:
            last_verbose_fe = fe_used
            print(
                f"  FE {fe_used:6d}/{FE_max}: fitness={global_best_fitness:.6f} "
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

    [FIX-7] Validates that *image* is a 2-D grayscale array.

    Parameters
    ----------
    image        : grayscale MRI image (H x W).  uint8 preferred; float
                   arrays are normalised to [0, 255] uint8 automatically.
    n_thresholds : number of threshold levels D (produces D+1 segments)
    hypso_kwargs : extra keyword arguments forwarded to hypso()
    verbose      : whether to print progress

    Returns
    -------
    (thresholds, segmented_image, best_fitness)
      thresholds      : 1-D sorted int array of shape (D,)
      segmented_image : H x W uint8 label map (values 0 .. D)
      best_fitness    : final 2-D Renyi entropy value
    """
    if image.ndim != 2:
        raise ValueError(
            f"image must be a 2-D grayscale array; got shape {image.shape}. "
            "For colour images, convert to grayscale first."
        )
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if n_thresholds < 1:
        raise ValueError("n_thresholds must be >= 1.")

    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        img = image.astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo) * 255.0
        image = img.clip(0, 255).astype(np.uint8)

    if verbose:
        print("[HyPSO] Computing nonlocal-means filtered image ...")
    filtered = nonlocal_means_filter(image)

    if verbose:
        print("[HyPSO] Building 2-D joint histogram ...")
    joint_hist = build_joint_histogram(image, filtered, num_bins=_NBINS)

    if verbose:
        print(f"[HyPSO] Running optimisation with D={n_thresholds} thresholds ...")
    kwargs = dict(hypso_kwargs or {})
    kwargs.setdefault("verbose", verbose)
    thresholds, fitness = hypso(joint_hist, D=n_thresholds, **kwargs)

    # Apply thresholds to produce integer label map
    boundaries = np.concatenate([[0], thresholds, [_NBINS]])
    segmented = np.zeros_like(image, dtype=np.uint8)
    for k in range(len(boundaries) - 1):
        lo, hi = int(boundaries[k]), int(boundaries[k + 1])
        segmented[(image >= lo) & (image < hi)] = k

    return thresholds, segmented, fitness


# ===========================================================================
# 8.  EVALUATION METRIC — SSIM  (reconstruction fidelity)
# ===========================================================================

def compute_ssim(image: np.ndarray, segmented: np.ndarray) -> float:
    """Structural Similarity Index between the original and the reconstructed
    image (label map mapped back to per-segment intensity means).

    This is the primary evaluation metric used in the paper.
    Requires scikit-image.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        raise ImportError(
            "scikit-image is required for SSIM computation. "
            "Install it with:  pip install scikit-image"
        )
    unique_labels = np.unique(segmented)
    reconstructed = np.zeros_like(image, dtype=np.float64)
    for lbl in unique_labels:
        mask = segmented == lbl
        reconstructed[mask] = float(image[mask].mean())
    reconstructed = reconstructed.clip(0, 255).astype(np.uint8)
    return float(ssim_fn(image, reconstructed, data_range=255))


# ===========================================================================
# 9.  COMMAND-LINE DEMO
# ===========================================================================

def _demo() -> None:
    """Quick self-test on a synthetic multi-region image."""
    print("=" * 60)
    print("HyPSO MRI Segmentation — Demo")
    print("=" * 60)

    rng = np.random.default_rng(42)
    h, w = 128, 128
    img = np.full((h, w), 30, dtype=np.uint8)
    img[20:60, 20:60] = 90
    img[60:100, 60:100] = 150
    img[10:30, 90:110] = 200
    noise = rng.integers(-10, 10, size=(h, w))
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    D = 3
    print(f"Image shape : {img.shape}, dtype: {img.dtype}")
    print(f"Thresholds  : D = {D}  →  {D + 1} segments")
    print()

    thresholds, segmented, fitness = segment_mri(
        img,
        n_thresholds=D,
        hypso_kwargs={
            "N": 20,
            "FE_max": 4_000,   # small budget for demo speed
            "seed": 1,
        },
        verbose=True,
    )

    print()
    print(f"Optimal thresholds : {thresholds.tolist()}")
    print(f"2D Renyi entropy   : {fitness:.6f}")
    print(f"Unique segments    : {np.unique(segmented).tolist()}")

    try:
        score = compute_ssim(img, segmented)
        print(f"SSIM               : {score:.4f}")
    except ImportError:
        print("SSIM: scikit-image not available, skipping.")

    print()
    print("Demo completed successfully.")


if __name__ == "__main__":
    _demo()
