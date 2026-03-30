# cd /Users/davisbone/Repositories/Bespoke-Ultrasound
# .venv/bin/python simulateTrajectories.py

"""
Synthetic Cell Trajectory Simulator
=====================================
Simulates underdamped Langevin trajectories of cells driven by the
acoustic radiation force fields produced by k-Wave (stored in training_data/).

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
    trajectory_data/<geometry>/traj_<sim_idx>_<cell_idx>.csv
    trajectory_data/traj_metadata.json

Usage:
    python simulate_trajectories.py
"""

import os
import sys
import json
import numpy as np
import h5py
from pathlib import Path

# ── SFI package (expects to be importable as 'SFI') ─────────────────────────
sys.path.insert(0, 'StochasticForceInference')
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from jax import random, jit
from SFI.SFI_Langevin import UnderdampedLangevinProcess

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

# ── Acoustic force scaling ───────────────────────────────────────────────────
# The k-Wave source was driven at 1 Pa (amplitude=1.0 in source.p).
# Acoustic radiation force ∝ I ∝ p_rms².
# Scale to the experimentally relevant LIPUS pressure (Harrison et al. 2025: ~160 kPa).
TARGET_PRESSURE_PA  = 160e3        # Pa — target acoustic pressure at the cell layer
KWAVE_SOURCE_PA     = 1.0          # Pa — source amplitude used in simulation
PRESSURE_SCALE      = (TARGET_PRESSURE_PA / KWAVE_SOURCE_PA) ** 2   # ≈ 2.56e10

# ── Cell biophysics ──────────────────────────────────────────────────────────
# Effective damping in the underdamped Langevin model.
# True Stokes drag for a 10 µm cell: γ = 6πηr/m ≈ 30,000 s⁻¹ (heavily overdamped).
# We use a much lower effective γ (1 s⁻¹) for a synthetic underdamped regime,
# matching the ULI algorithm's intent and giving visible inertia in the trajectories.
# This mimics "persistent" active cell motility, which the ULI framework is designed for.
GAMMA = 1.0           # effective damping  [s⁻¹]

# Diffusion in velocity space (D), calibrated so that the acoustic drift competes
# with the stochastic velocity fluctuations at SNR ≈ 3:
#   v_thermal ~ sqrt(D/γ) ≈ 1.5e-7 m/s
#   v_terminal = F_max/γ  ≈ 4e-7 m/s   (at 160 kPa, FORCE_SCALE below)
DIFFUSION = 1e-13     # D [m²/s³]  — velocity-space diffusion (NOT positional)

# Cell geometry (used only to convert radiation force density → acceleration)
CELL_DENSITY_KG_M3 = 1050.0   # kg/m³  (slightly denser than water)
CELL_RADIUS_M      = 10e-6    # m      (10 µm radius)
_cell_volume = (4.0 / 3.0) * np.pi * CELL_RADIUS_M ** 3   # m³  ≈ 4.2e-15
# Force density [N/m³] / cell_density [kg/m³] = acceleration [m/s²]
FORCE_SCALE = 1.0 / CELL_DENSITY_KG_M3   # [m³/kg]

# Trajectory parameters
N_CELLS       = 5           # independent cells per acoustic field
N_STEPS       = 5_000       # recorded time steps
DT            = 0.01        # s — recording interval (10 ms)
OVERSAMPLING  = 10          # integration substeps per DT  →  ddt = 1 ms
PRERUN        = 200         # equilibration steps (not recorded)

# Training data directory
TRAINING_DATA_ROOT = 'trainingData'
OUTPUT_ROOT        = 'trajectoryData'
PML_SIZE           = 20     # must match kwave_training_data_generator.py


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: load k-Wave radiation force field and build interpolator
# ──────────────────────────────────────────────────────────────────────────────

def load_force_field(h5_path: str):
    """
    Load the radiation force density field from HDF5.

    Scales to TARGET_PRESSURE_PA and converts N/m³ → m/s² (acceleration).
    Returns the raw grid arrays for JAX-native bilinear interpolation.

    Returns
    -------
    Fy_grid : np.ndarray  shape (iNx, iNy), downward force in m/s²
    x_m     : np.ndarray  horizontal coordinates [m]
    y_m     : np.ndarray  depth coordinates      [m]
    """
    with h5py.File(h5_path, 'r') as f:
        rf = f['radiation_force_dens'][:]   # shape (Nx, Ny), N/m³
        Nx  = int(f.attrs['Nx'])
        Ny  = int(f.attrs['Ny'])
        dx  = float(f.attrs['dx_m'])

    # Trim PML padding
    rf_interior = rf[PML_SIZE:Nx - PML_SIZE, PML_SIZE:Ny - PML_SIZE]

    # Physical coordinate arrays (origin = top-left of interior domain)
    iNx, iNy = rf_interior.shape
    x_m = np.arange(iNx, dtype=np.float32) * dx
    y_m = np.arange(iNy, dtype=np.float32) * dx

    # Scale to target LIPUS pressure (RF ∝ p²) then convert N/m³ → m/s²
    Fy_grid = (rf_interior * PRESSURE_SCALE * FORCE_SCALE).astype(np.float32)

    return Fy_grid, x_m, y_m


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
    # Convert grid to JAX arrays (captured in closure, not traced)
    x_jax  = jnp.array(x_m, dtype=jnp.float32)
    y_jax  = jnp.array(y_m, dtype=jnp.float32)
    Fy_jax = jnp.array(Fy_grid, dtype=jnp.float32)  # shape (iNx, iNy)

    dx = x_jax[1] - x_jax[0]
    dy = y_jax[1] - y_jax[0]
    x0 = x_jax[0]
    y0 = y_jax[0]
    Nx_g = x_jax.shape[0]
    Ny_g = y_jax.shape[0]

    def bilinear(x, y):
        """Bilinear interpolation of Fy_jax at (x, y)."""
        # Clamp
        xc = jnp.clip(x, x0, x_jax[-1])
        yc = jnp.clip(y, y0, y_jax[-1])
        # Grid indices (integer)
        ix = jnp.floor((xc - x0) / dx).astype(jnp.int32)
        iy = jnp.floor((yc - y0) / dy).astype(jnp.int32)
        ix = jnp.clip(ix, 0, Nx_g - 2)
        iy = jnp.clip(iy, 0, Ny_g - 2)
        # Fractional offsets
        tx = (xc - (x0 + ix * dx)) / dx
        ty = (yc - (y0 + iy * dy)) / dy
        # Four corners
        f00 = Fy_jax[ix,     iy    ]
        f10 = Fy_jax[ix + 1, iy    ]
        f01 = Fy_jax[ix,     iy + 1]
        f11 = Fy_jax[ix + 1, iy + 1]
        return ((1 - tx) * (1 - ty) * f00
              + tx       * (1 - ty) * f10
              + (1 - tx) * ty       * f01
              + tx       * ty       * f11)

    def force(X, V, theta):
        gamma = theta[0]
        # Acoustic body force acts downward (y-direction only)
        F_acoustic = jnp.array([0.0, bilinear(X[0], X[1])])
        return F_acoustic - gamma * V

    return force


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATE ONE ACOUSTIC FIELD
# ──────────────────────────────────────────────────────────────────────────────

def simulate_for_field(h5_path: str, sim_idx: int, output_dir: str, key):
    """
    Simulate N_CELLS cell trajectories driven by the acoustic field in h5_path.
    Saves one CSV per cell, returns list of output paths.
    """
    Fy_grid, x_m, y_m = load_force_field(h5_path)
    x_min, x_max = float(x_m[0]), float(x_m[-1])
    y_min, y_max = float(y_m[0]), float(y_m[-1])
    force_fn = make_force_fn(Fy_grid, x_m, y_m)

    theta_F = jnp.array([GAMMA])   # params passed to force_fn
    D_matrix = DIFFUSION * jnp.eye(2)

    output_paths = []
    for cell_idx in range(N_CELLS):
        key, subkey = random.split(key)

        # Random initial position inside the domain (avoid walls)
        margin = 0.05
        x0 = np.random.uniform(x_min + margin * (x_max - x_min),
                                x_max - margin * (x_max - x_min))
        y0 = np.random.uniform(y_min + margin * (y_max - y_min),
                                y_max - margin * (y_max - y_min))
        init_pos = jnp.array([x0, y0])
        init_vel = jnp.zeros(2)

        model = UnderdampedLangevinProcess(force_fn, D_matrix)
        model.initialize(init_pos, init_vel, params_F=theta_F)
        model.simulate(DT, N_STEPS, subkey,
                       oversampling=OVERSAMPLING, prerun=PRERUN)

        fname = os.path.join(output_dir, f"traj_{sim_idx:04d}_cell{cell_idx:02d}.csv")
        model.save_trajectory_data(fname)
        output_paths.append(fname)
        print(f"    Cell {cell_idx:02d} → {fname}")

    return output_paths, key


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    meta_path = os.path.join(TRAINING_DATA_ROOT, 'metadata.json')
    with open(meta_path) as f:
        all_sim_meta = json.load(f)

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
        traj_paths, key = simulate_for_field(h5_path, sim_idx, geo_out, key)

        traj_metadata.append({
            'sim_index': sim_idx,
            'geometry': geometry,
            'n_transducers': sim_meta['n_transducers'],
            'n_cells': N_CELLS,
            'dt': DT,
            'n_steps': N_STEPS,
            'oversampling': OVERSAMPLING,
            'gamma': GAMMA,
            'diffusion': DIFFUSION,
            'trajectory_files': traj_paths,
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
    print(f"Cells per field:  {N_CELLS}")
    print(f"Steps per traj:   {N_STEPS}  ×  dt={DT:.3f} s  = {N_STEPS*DT:.1f} s total")
    print(f"Damping γ:        {GAMMA} s⁻¹")
    print(f"Diffusion D:      {DIFFUSION:.1e} m²/s")
    print(f"Oversampling:     {OVERSAMPLING}×")
    print()
    main()
