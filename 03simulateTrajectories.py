# cd /Users/davisbone/Repositories/Bespoke-Ultrasound
# .venv/bin/python 03simulateTrajectories.py

"""
Synthetic Cell Trajectory Simulator
=====================================
Simulates underdamped Langevin trajectories of cells driven by the
acoustic radiation force fields produced by k-Wave (stored in trainingData/).

Physics model (underdamped Langevin):
    dx/dt = v
    dv/dt = F_acoustic(x) - γ·v + σ·ξ(t)

where:
    F_acoustic(x) : radiation force interpolated from the k-Wave HDF5 field  [m/s²]
    γ              : viscous damping coefficient  [s⁻¹]
    σ              : noise amplitude  [m/s^(3/2)]
    ξ(t)           : Gaussian white noise

Units used throughout:
    position   : metres  [m]
    velocity   : m/s
    force/accel: m/s²
    time       : seconds [s]

Output (per simulation):
    trajectoryData/<geometry>/traj_<sim_idx>.csv   (all cells, multi-particle CSV)
    trajectoryData/traj_metadata.json

Usage:
    # Simulate all acoustic fields
    python 03simulateTrajectories.py

    # Simulate at most N fields (useful for development/testing)
    python 03simulateTrajectories.py --max 5

    # Simulate one geometry only
    python 03simulateTrajectories.py --geometry well
"""

import argparse
import os
import sys
import json
import numpy as np
import h5py
from pathlib import Path

# ── SFI package (expects to be importable as 'SFI') ─────────────────────────
sys.path.insert(0, 'StochasticForceInference')
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)   # float64 needed: step sizes ~3 nm, float32 resolution ~120 nm
import jax.numpy as jnp
from jax import random, jit
from SFI.SFI_Langevin import UnderdampedLangevinProcess

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# Medium sound speed — must match kwaveTrainingDataGenerator.py
MEDIUM_SOUND_SPEED = 1540.0   # m/s

# ── Acoustic force scaling ───────────────────────────────────────────────────
# The k-Wave source was driven at amplitude=1.0 via tone_burst (≈ 1 Pa peak).
# Acoustic radiation force ∝ I ∝ p_rms².
# Scale to the experimentally relevant LIPUS pressure (Harrison et al. 2025: ~160 kPa).
TARGET_PRESSURE_PA  = 160e3        # Pa — target acoustic pressure at the cell layer
KWAVE_SOURCE_PA     = 1.0          # Pa — source amplitude used in simulation
PRESSURE_SCALE      = (TARGET_PRESSURE_PA / KWAVE_SOURCE_PA) ** 2   # ≈ 2.56e10

# Synthetic force boost for ULI demonstration.
# radiation_force_dens is near-zero for the nearly lossless medium (see kwaveTrainingDataGenerator).
# We instead derive ARF from intensity using the lossless-limit formula F = 2I/c [N/m³],
# which preserves the correct spatial beam pattern.
# FORCE_BOOST amplifies the pattern to a regime where ULI inference is tractable;
# it does NOT change the force topology (beam pattern), only its magnitude.
# Calibrated so that v_terminal = F_max/γ ≈ 8e-5 m/s → ~4 mm drift over 50 s (18% of well).
FORCE_BOOST = 200     # dimensionless; effective force ~4e-3 m/s² after boost

# ── Cell biophysics ──────────────────────────────────────────────────────────
# Synthetic damping tuned so that:
#   γ · DT = 50 × 0.01 = 0.5  →  intermediate regime optimal for ULI WeakNoise estimator
#   (γ·DT << 1 is underdamped but biases the D estimator; γ·DT >> 1 is overdamped and
#    the ULI force signal (velocity change) is dominated by noise rather than force)
# True Stokes drag: γ = 6πηr/m ≈ 30,000 s⁻¹; here chosen for algorithmic convenience.
GAMMA = 50.0          # effective damping  [s⁻¹]

# Diffusion in velocity space (D), calibrated so that the acoustic drift competes
# with the stochastic velocity fluctuations at SNR ≈ 6:
#   v_thermal ~ sqrt(D/γ) = sqrt(1e-8/50) ≈ 1.4e-5 m/s
#   v_terminal = F_max/γ  = 4e-3/50       = 8e-5 m/s   →  SNR ≈ 6
#   drift in 50 s         = 8e-5 × 50     = 4 mm (18 % of well width)
DIFFUSION = 1e-8      # D [m²/s³]  — velocity-space diffusion (NOT positional)

# Cell geometry (used only to convert radiation force density → acceleration)
CELL_DENSITY_KG_M3 = 1050.0   # kg/m³  (slightly denser than water)
CELL_RADIUS_M      = 10e-6    # m      (10 µm radius)
_cell_volume = (4.0 / 3.0) * np.pi * CELL_RADIUS_M ** 3   # m³  ≈ 4.2e-15
# Force density [N/m³] / cell_density [kg/m³] = acceleration [m/s²]
FORCE_SCALE = 1.0 / CELL_DENSITY_KG_M3   # [m³/kg]

# Trajectory parameters
# N_CELLS is a target — actual count = nx × ny, chosen to match domain aspect ratio.
# 50 cells on a regular grid gives ULI enough spatial coverage to fit the beam pattern.
N_CELLS       = 50          # target cells per acoustic field
N_STEPS       = 5_000       # recorded time steps
DT            = 0.01        # s — recording interval (10 ms)
OVERSAMPLING  = 10          # integration substeps per DT  →  ddt = 1 ms
PRERUN        = 200         # equilibration steps (not recorded)

# Training data directory
TRAINING_DATA_ROOT = 'trainingData'
OUTPUT_ROOT        = 'trajectoryData'
PML_SIZE           = 20     # must match kwaveTrainingDataGenerator.py


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: load k-Wave radiation force field
# ──────────────────────────────────────────────────────────────────────────────

def load_force_field(h5_path: str):
    """
    Load the acoustic intensity field from HDF5 and derive a radiation force proxy.

    ARF is estimated using the lossless-limit formula F = 2·I/c [N/m³], which
    gives a physically meaningful spatial pattern even when the medium absorption
    is near-zero (where the stored radiation_force_dens field would be ~0).

    Scales to TARGET_PRESSURE_PA and converts N/m³ → m/s² (acceleration).
    Returns raw grid arrays for JAX-native bilinear interpolation.

    Returns
    -------
    Fy_grid : np.ndarray  shape (iNx, iNy), downward force in m/s²
    x_m     : np.ndarray  horizontal coordinates [m]
    y_m     : np.ndarray  depth coordinates      [m]
    """
    with h5py.File(h5_path, 'r') as f:
        intensity = np.array(f['intensity'])
        Nx = int(np.asarray(f.attrs['Nx']).item())
        Ny = int(np.asarray(f.attrs['Ny']).item())
        dx = float(np.asarray(f.attrs['dx_m']).item())

    intensity_interior = intensity[PML_SIZE:Nx - PML_SIZE, PML_SIZE:Ny - PML_SIZE]
    iNx, iNy = intensity_interior.shape
    x_m = np.arange(iNx, dtype=np.float64) * dx
    y_m = np.arange(iNy, dtype=np.float64) * dx

    # F [N/m³] = 2·I/c  (lossless limit radiation force density)
    rf_proxy = intensity_interior * (2.0 / MEDIUM_SOUND_SPEED)
    Fy_grid = (rf_proxy * PRESSURE_SCALE * FORCE_SCALE * FORCE_BOOST).astype(np.float64)
    return Fy_grid, x_m, y_m


# ──────────────────────────────────────────────────────────────────────────────
# GRID PLACEMENT
# ──────────────────────────────────────────────────────────────────────────────

def make_grid_positions(x_m: np.ndarray, y_m: np.ndarray,
                        n_cells: int, margin: float = 0.08):
    """
    Return a regular grid of ~n_cells starting positions within the domain.

    The grid aspect ratio matches the domain so cells are roughly uniformly
    spaced regardless of geometry.

    Returns
    -------
    positions : np.ndarray  shape (n_actual, 2)  in metres
    """
    x_span = float(x_m[-1] - x_m[0])
    y_span = float(y_m[-1] - y_m[0])
    aspect = x_span / y_span

    ny = max(2, int(round(np.sqrt(n_cells / aspect))))
    nx = max(2, int(round(n_cells / ny)))

    x_lo = float(x_m[0])  + margin * x_span
    x_hi = float(x_m[-1]) - margin * x_span
    y_lo = float(y_m[0])  + margin * y_span
    y_hi = float(y_m[-1]) - margin * y_span

    xs = np.linspace(x_lo, x_hi, nx)
    ys = np.linspace(y_lo, y_hi, ny)
    XX, YY = np.meshgrid(xs, ys)
    positions = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float64)
    return positions, nx, ny


# ──────────────────────────────────────────────────────────────────────────────
# FORCE FUNCTION for UnderdampedLangevinProcess
# ──────────────────────────────────────────────────────────────────────────────

def make_force_fn(Fy_grid: np.ndarray, x_m: np.ndarray, y_m: np.ndarray):
    """
    Return a JAX-traceable force function for one cell in 2-D.

    State:  X = [x, y]  (position in metres)
            V = [vx, vy] (velocity in m/s)
    Params: theta = [gamma]  (viscous damping, s⁻¹)

    Force = F_acoustic(x,y) · ŷ  −  γ · V

    Uses JAX-native bilinear interpolation so the function can be
    JIT-compiled and vectorised by the SFI simulation engine.
    """
    x_jax  = jnp.array(x_m, dtype=jnp.float64)
    y_jax  = jnp.array(y_m, dtype=jnp.float64)
    Fy_jax = jnp.array(Fy_grid, dtype=jnp.float64)

    dx = x_jax[1] - x_jax[0]
    dy = y_jax[1] - y_jax[0]
    x0 = x_jax[0]
    y0 = y_jax[0]
    Nx_g = x_jax.shape[0]
    Ny_g = y_jax.shape[0]

    def bilinear(x, y):
        xc = jnp.clip(x, x0, x_jax[-1])
        yc = jnp.clip(y, y0, y_jax[-1])
        ix = jnp.clip(jnp.floor((xc - x0) / dx).astype(jnp.int32), 0, Nx_g - 2)
        iy = jnp.clip(jnp.floor((yc - y0) / dy).astype(jnp.int32), 0, Ny_g - 2)
        tx = (xc - (x0 + ix * dx)) / dx
        ty = (yc - (y0 + iy * dy)) / dy
        return ((1 - tx) * (1 - ty) * Fy_jax[ix,     iy    ]
              + tx       * (1 - ty) * Fy_jax[ix + 1, iy    ]
              + (1 - tx) * ty       * Fy_jax[ix,     iy + 1]
              + tx       * ty       * Fy_jax[ix + 1, iy + 1])

    def force(X, V, theta):
        gamma = theta[0]
        F_acoustic = jnp.array([0.0, bilinear(X[0], X[1])])
        return F_acoustic - gamma * V

    return force


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATE ONE ACOUSTIC FIELD
# ──────────────────────────────────────────────────────────────────────────────

def simulate_for_field(h5_path: str, sim_idx: int, output_dir: str, key):
    """
    Simulate all cells for one acoustic field in a SINGLE multi-particle call.

    All cells share the same compiled force function — compilation happens once
    per acoustic field, not once per cell.

    Saves one multi-particle CSV (all cells, particle_id column distinguishes them).
    Returns the output path and updated PRNG key.
    """
    Fy_grid, x_m, y_m = load_force_field(h5_path)
    force_fn = make_force_fn(Fy_grid, x_m, y_m)

    # Place cells on a regular grid
    positions, nx, ny = make_grid_positions(x_m, y_m, N_CELLS)
    n_actual = len(positions)

    init_pos = jnp.array(positions)          # (n_actual, 2)
    init_vel = jnp.zeros((n_actual, 2))
    theta_F  = jnp.array([GAMMA])
    D_matrix = DIFFUSION * jnp.eye(2)

    key, subkey = random.split(key)
    model = UnderdampedLangevinProcess(force_fn, D_matrix)
    model.initialize(init_pos, init_vel, params_F=theta_F)
    model.simulate(DT, N_STEPS, subkey,
                   oversampling=OVERSAMPLING, prerun=PRERUN)

    fname = os.path.join(output_dir, f"traj_{sim_idx:04d}.csv")
    model.save_trajectory_data(fname)
    print(f"    {n_actual} cells ({nx}×{ny} grid) → {fname}")

    return fname, n_actual, key


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Simulate underdamped cell trajectories')
    parser.add_argument('--max', type=int, metavar='N',
                        help='Simulate at most N acoustic fields (per geometry if --geometry set)')
    parser.add_argument('--geometry', choices=['well', 'slide'],
                        help='Restrict to one geometry only')
    args = parser.parse_args()

    meta_path = os.path.join(TRAINING_DATA_ROOT, 'metadata.json')
    with open(meta_path) as f:
        all_sim_meta = json.load(f)

    # Apply geometry filter then max-count limit
    if args.geometry:
        all_sim_meta = [m for m in all_sim_meta if m['geometry'] == args.geometry]
    if args.max:
        all_sim_meta = all_sim_meta[:args.max]

    Path(OUTPUT_ROOT).mkdir(exist_ok=True)

    key = random.PRNGKey(42)
    traj_metadata = []

    for sim_meta in all_sim_meta:
        sim_idx  = sim_meta['sim_index']
        geometry = sim_meta['geometry']
        h5_path  = sim_meta['file']

        geo_out = os.path.join(OUTPUT_ROOT, geometry)
        Path(geo_out).mkdir(exist_ok=True)

        print(f"\nSim {sim_idx:04d}  [{geometry}]  {sim_meta['n_transducers']} transducer(s)")
        traj_path, n_cells, key = simulate_for_field(h5_path, sim_idx, geo_out, key)

        traj_metadata.append({
            'sim_index': sim_idx,
            'geometry': geometry,
            'n_transducers': sim_meta['n_transducers'],
            'n_cells': n_cells,
            'dt': DT,
            'n_steps': N_STEPS,
            'oversampling': OVERSAMPLING,
            'gamma': GAMMA,
            'diffusion': DIFFUSION,
            'force_boost': FORCE_BOOST,
            'trajectory_file': traj_path,
            'acoustic_field_file': h5_path,
        })

    out_meta = os.path.join(OUTPUT_ROOT, 'traj_metadata.json')
    with open(out_meta, 'w') as f:
        json.dump(traj_metadata, f, indent=2)

    print(f"\n✓ Trajectories saved to {OUTPUT_ROOT}/")
    print(f"  Metadata: {out_meta}")


if __name__ == '__main__':
    print("Underdamped Cell Trajectory Simulator")
    print("=" * 50)
    print(f"Target cells per field: {N_CELLS}  (regular grid)")
    print(f"Steps per traj:         {N_STEPS}  ×  dt={DT:.3f} s  = {N_STEPS*DT:.1f} s total")
    print(f"Damping γ:              {GAMMA} s⁻¹")
    print(f"Diffusion D:            {DIFFUSION:.1e} m²/s³")
    print(f"Oversampling:           {OVERSAMPLING}×")
    print()
    main()
