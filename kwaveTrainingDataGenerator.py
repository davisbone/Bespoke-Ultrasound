# cd /Users/davisbone/Repositories/Bespoke-Ultrasound
# .venv/bin/python kwaveTrainingDataGenerator.py

"""
Acoustic Pressure Field Training Data Generator
================================================
Generates k-Wave simulation training data for acoustic pressure fields
in two geometries:
  1. A single well of a 12-well plate (cylindrical cross-section)
  2. A flat rectangular microscope slide

Goal: Simulate multiple transducer configurations that approximate a
"gravity-like" (uniform downward) acoustic radiation force field, and
save pressure field snapshots as training data for ULI inference.

Requirements:
    pip install k-wave-python numpy h5py matplotlib tqdm

Usage:
    python kwaveTrainingDataGenerator.py

Output:
    trainingData/
        well_plate/         # .h5 files, one per simulation
        microscope_slide/   # .h5 files, one per simulation
        metadata.json       # all simulation parameters
"""

import numpy as np
import h5py
import json
import os
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal

# k-wave-python imports
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.signals import tone_burst
from kwave.utils.mapgen import make_disc

# ──────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS & GEOMETRY DEFINITIONS
# ──────────────────────────────────────────────────────────────────

# Tissue/culture medium properties (cell culture media ≈ water at 37°C)
MEDIUM_SOUND_SPEED = 1540.0   # m/s
MEDIUM_DENSITY     = 1000.0   # kg/m³
MEDIUM_ALPHA_COEFF = 0.002    # dB/(MHz^y cm), nearly lossless for water

# Simulation frequency (1 MHz standard LIPUS)
FREQUENCY = 1.0e6  # Hz

# Points per wavelength (higher = more accurate but slower; 6–10 is typical)
PPW = 8

# Compute grid spacing from PPW
DX = MEDIUM_SOUND_SPEED / (PPW * FREQUENCY)  # ~0.19 mm at 1 MHz, 8 PPW

# CFL number for stable time stepping
CFL = 0.3

# ──────────────────────────────────────────────────────────────────
# GEOMETRY: 12-WELL PLATE (single well)
# ──────────────────────────────────────────────────────────────────
# Standard 12-well plate: ~22.1 mm inner diameter, ~17 mm fluid depth
WELL_DIAMETER_M   = 0.0221   # m
WELL_DEPTH_M      = 0.007    # m  (typical seeding depth ~7 mm)

# ──────────────────────────────────────────────────────────────────
# GEOMETRY: MICROSCOPE SLIDE
# ──────────────────────────────────────────────────────────────────
# Standard microscope slide region of interest
SLIDE_WIDTH_M  = 0.025  # m  (25 mm ROI)
SLIDE_DEPTH_M  = 0.003  # m  (3 mm thin liquid layer)

# Perfectly matched layer (absorbing boundary) thickness in grid points
PML_SIZE = 20


@dataclass
class TransducerConfig:
    """Defines a single transducer: position (normalized 0–1), phase offset."""
    x_norm: float   # normalized horizontal position (0 = left wall, 1 = right wall)
    phase_deg: float = 0.0
    amplitude: float = 1.0  # relative amplitude (1 = full)


@dataclass
class SimulationParams:
    """All parameters for one simulation run."""
    geometry: Literal["well", "slide"]
    transducers: list
    frequency_hz: float
    ppw: int
    cfl: float
    medium_sound_speed: float
    medium_density: float
    sim_index: int


def make_grid(geometry: str):
    """
    Create kWaveGrid for the chosen geometry.
    Returns (kgrid, domain_size_m) where domain_size_m = (Nx, Ny) in meters.
    
    Convention: x = horizontal, y = vertical (depth, gravity direction)
    Transducers fire from the TOP (y=0 wall), cells sit at y=Ny (bottom).
    """
    if geometry == "well":
        width_m = WELL_DIAMETER_M
        depth_m = WELL_DEPTH_M
    elif geometry == "slide":
        width_m = SLIDE_WIDTH_M
        depth_m = SLIDE_DEPTH_M
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    Nx = int(np.ceil(width_m / DX))
    Ny = int(np.ceil(depth_m / DX))

    # Include PML in grid dimensions (pml_inside=True: PML is carved from the grid)
    Nx += 2 * PML_SIZE
    Ny += 2 * PML_SIZE

    # Ensure even grid dimensions (k-Wave requirement)
    Nx = Nx + (Nx % 2)
    Ny = Ny + (Ny % 2)

    kgrid = kWaveGrid([Nx, Ny], [DX, DX])
    return kgrid, (Nx, Ny)


def make_medium():
    """Homogeneous aqueous medium (cell culture media)."""
    medium = kWaveMedium(
        sound_speed=MEDIUM_SOUND_SPEED,
        density=MEDIUM_DENSITY,
        alpha_coeff=MEDIUM_ALPHA_COEFF,
        alpha_power=2.0,
    )
    return medium


def place_transducers(kgrid, Nx: int, Ny: int, configs: list[TransducerConfig]):
    """
    Place linear transducer elements along the TOP wall (y = PML_SIZE + 1).
    Each transducer is a single-grid-point source (point source approximation).
    For a real array, extend this to a line segment source.

    Returns a kSource object with combined pressure signal.
    """
    source = kSource()

    # Transducer y-position: just inside the PML at the top
    y_src = PML_SIZE + 1

    # Build a binary mask and a signal array
    source_mask = np.zeros((Nx, Ny), dtype=bool)
    source_x_indices = []

    for cfg in configs:
        x_idx = int(np.clip(cfg.x_norm * Nx, 1, Nx - 2))
        source_mask[x_idx, y_src] = True
        source_x_indices.append((x_idx, cfg.phase_deg, cfg.amplitude))

    source.p_mask = source_mask.astype(int)

    # Build tone burst signals for each source point
    t_end = 30.0 / FREQUENCY        # 30 cycles — enough for steady-state RMS
    dt = CFL * DX / MEDIUM_SOUND_SPEED
    t_array = np.arange(0, t_end, dt)
    n_t = len(t_array)

    num_sources = int(source_mask.sum())
    source_signals = np.zeros((num_sources, n_t))

    # Map source points in mask order (k-Wave reads mask column-major)
    mask_flat = source_mask.flatten(order='F')
    active_flat_indices = np.where(mask_flat)[0]

    for flat_idx_pos, (x_idx, phase_deg, amplitude) in enumerate(source_x_indices):
        # Flat index in column-major order
        flat_idx = y_src * Nx + x_idx  # row-major approximation; adjust if needed
        # Find position in active sources list
        try:
            src_pos = list(active_flat_indices).index(
                np.ravel_multi_index([x_idx, y_src], (Nx, Ny), order='F')
            )
        except ValueError:
            src_pos = flat_idx_pos

        phase_rad = np.deg2rad(phase_deg)
        signal = amplitude * np.sin(2 * np.pi * FREQUENCY * t_array + phase_rad)
        # Apply Hanning taper to reduce spectral leakage
        taper = np.hanning(n_t)
        source_signals[src_pos % num_sources, :] = signal * taper

    source.p = source_signals
    return source, dt, n_t


def make_sensor(Nx: int, Ny: int):
    """
    Record pressure across the entire bottom plane (cell layer) and
    a 2D snapshot mask across the full domain for training data.
    """
    sensor = kSensor()
    # Full-domain pressure map (for training data)
    sensor.mask = np.ones((Nx, Ny), dtype=int)
    sensor.record = ['p']
    return sensor


def build_gravity_like_configs(geometry: str, n_transducers_list: list[int]):
    """
    Generate transducer configurations designed to create a net downward
    acoustic radiation force — analogous to a gravity-like field.

    Strategy: evenly spaced transducers across the top wall, all in-phase.
    Variations include:
      - Number of transducers (1, 2, 4, 8, ...)
      - Phase gradients to steer the beam angle (small angles → near-vertical)
      - Amplitude tapering (uniform vs. Gaussian apodization)

    Returns a list of lists of TransducerConfig.
    """
    all_configs = []

    for n in n_transducers_list:
        if n == 1:
            # Single centered transducer
            cfgs = [TransducerConfig(x_norm=0.5, phase_deg=0.0)]
            all_configs.append(cfgs)
        else:
            # Evenly spaced, in-phase (pure downward)
            positions = np.linspace(0.1, 0.9, n)

            # Config 1: All in-phase (uniform downward)
            cfgs_uniform = [TransducerConfig(x_norm=p, phase_deg=0.0) for p in positions]
            all_configs.append(cfgs_uniform)

            # Config 2: Gaussian amplitude taper (reduce edge effects)
            center = 0.5
            sigma = 0.25
            weights = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
            cfgs_gauss = [
                TransducerConfig(x_norm=p, phase_deg=0.0, amplitude=float(w))
                for p, w in zip(positions, weights)
            ]
            all_configs.append(cfgs_gauss)

            # Config 3: Small linear phase gradient (slight beam steering ~5°)
            # Phase shift per element: Δφ = 2π * d * sin(θ) / λ
            d = (0.9 - 0.1) / (n - 1) * (WELL_DIAMETER_M if geometry == "well" else SLIDE_WIDTH_M)
            wavelength = MEDIUM_SOUND_SPEED / FREQUENCY
            theta_deg = 5.0
            delta_phi = 360.0 * d * np.sin(np.deg2rad(theta_deg)) / wavelength
            phases = np.arange(n) * delta_phi
            cfgs_steered = [
                TransducerConfig(x_norm=p, phase_deg=float(ph))
                for p, ph in zip(positions, phases)
            ]
            all_configs.append(cfgs_steered)

    return all_configs


def run_simulation(geometry: str, configs: list[TransducerConfig], sim_idx: int,
                   output_dir: str):
    """
    Run one k-Wave simulation and save results to HDF5.
    Returns path to saved file.
    """
    kgrid, (Nx, Ny) = make_grid(geometry)
    medium = make_medium()

    source, dt, n_t = place_transducers(kgrid, Nx, Ny, configs)
    sensor = make_sensor(Nx, Ny)

    # Set time array explicitly
    kgrid.setTime(n_t, dt)

    sim_options = SimulationOptions(
        pml_size=PML_SIZE,
        save_to_disk=True,
        data_cast='single',
    )
    exec_options = SimulationExecutionOptions(is_gpu_simulation=False)

    print(f"  Running {geometry} sim {sim_idx:04d} | "
          f"{Nx}×{Ny} grid | {len(configs)} transducers | {n_t} timesteps")

    sensor_data = kspaceFirstOrder2D(
        kgrid, source, sensor, medium,
        simulation_options=sim_options,
        execution_options=exec_options,
    )
    assert sensor_data is not None

    # RMS pressure and final snapshot from time series (Nx*Ny × n_t)
    p_time = sensor_data['p']
    p_3d   = p_time.reshape(Nx, Ny, -1, order='F')
    p_rms  = np.sqrt(np.mean(p_3d ** 2, axis=-1))  # (Nx, Ny)
    p_final = p_3d[:, :, -1]                        # (Nx, Ny) last timestep

    # Acoustic intensity I = p_rms² / (ρ c)  [W/m²]
    intensity = p_rms ** 2 / (MEDIUM_DENSITY * MEDIUM_SOUND_SPEED)

    # Acoustic radiation force density (simplified): F = 2α*I/c  [N/m³]
    # For lossless media, use gradient of acoustic energy density instead
    alpha_np_m = MEDIUM_ALPHA_COEFF * 100 / (8.686 * FREQUENCY / 1e6)  # Np/m
    radiation_force_density = 2 * alpha_np_m * intensity / MEDIUM_SOUND_SPEED

    # Save to HDF5
    fname = os.path.join(output_dir, f"sim_{sim_idx:04d}.h5")
    with h5py.File(fname, 'w') as f:
        f.create_dataset('p_final',              data=p_final.astype(np.float32))
        f.create_dataset('p_rms',                data=p_rms.astype(np.float32))
        f.create_dataset('intensity',            data=intensity.astype(np.float32))
        f.create_dataset('radiation_force_dens', data=radiation_force_density.astype(np.float32))

        # Store grid metadata
        f.attrs['geometry']    = geometry
        f.attrs['Nx']          = Nx
        f.attrs['Ny']          = Ny
        f.attrs['dx_m']        = DX
        f.attrs['frequency_hz'] = FREQUENCY
        f.attrs['n_transducers'] = len(configs)

        # Store transducer positions and phases
        f.create_dataset('transducer_x_norm',   data=np.array([c.x_norm    for c in configs]))
        f.create_dataset('transducer_phase_deg', data=np.array([c.phase_deg for c in configs]))
        f.create_dataset('transducer_amplitude', data=np.array([c.amplitude for c in configs]))

    return fname, {
        'sim_index': sim_idx,
        'geometry': geometry,
        'n_transducers': len(configs),
        'transducers': [asdict(c) for c in configs],
        'Nx': Nx,
        'Ny': Ny,
        'dx_m': DX,
        'file': fname,
    }


def generate_training_dataset(
    geometries: list[str] = ['well', 'slide'],
    n_transducers_list: list[int] = [1, 2, 4, 8],
    output_root: str = 'trainingData',
):
    """
    Main entry point. Loops over geometries and transducer configurations,
    runs simulations, and saves results + metadata.
    """
    Path(output_root).mkdir(exist_ok=True)
    all_metadata = []
    sim_idx = 0

    for geometry in geometries:
        geo_dir = os.path.join(output_root, geometry)
        Path(geo_dir).mkdir(exist_ok=True)

        configs_list = build_gravity_like_configs(geometry, n_transducers_list)
        print(f"\n{'='*60}")
        print(f"Geometry: {geometry.upper()}")
        print(f"Total configurations: {len(configs_list)}")
        print(f"{'='*60}")

        for configs in configs_list:
            _, meta = run_simulation(geometry, configs, sim_idx, geo_dir)
            all_metadata.append(meta)
            sim_idx += 1

    # Write metadata JSON
    meta_path = os.path.join(output_root, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n✓ Completed {sim_idx} simulations.")
    print(f"  Metadata saved to {meta_path}")
    return all_metadata


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("k-Wave Acoustic Field Training Data Generator")
    print("=" * 50)
    print(f"Grid spacing:  {DX*1e3:.3f} mm  ({PPW} PPW at {FREQUENCY/1e6:.1f} MHz)")
    print(f"Medium:        {MEDIUM_SOUND_SPEED} m/s, {MEDIUM_DENSITY} kg/m³")
    print()

    metadata = generate_training_dataset(
        geometries=['well', 'slide'],
        n_transducers_list=[1, 2, 4, 8],
        output_root='trainingData',
    )
