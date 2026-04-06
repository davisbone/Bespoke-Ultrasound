# cd /Users/davisbone/Repositories/Bespoke-Ultrasound
# .venv/bin/python 02visualizeTrainingData.py

"""
Visualization of k-Wave acoustic pressure field training data.

For each simulation plots a three-panel figure:
  - RMS pressure field
  - Acoustic intensity field  (I = p_rms² / ρc)
  - Depth profile of RMS pressure at the horizontal centre
    (flat profile → good gravity-like uniformity)

NOTE: radiation_force_dens is stored in the HDF5 files but is physically
unreliable for this nearly-lossless medium (alpha ≈ 0.002 dB/(MHz·cm)),
so it is not plotted. Use p_rms / intensity for downstream ULI targets.

Usage:
    # Plot one representative sim per (geometry, transducer count) — fast default
    python 02visualizeTrainingData.py

    # Plot every simulation
    python 02visualizeTrainingData.py --all

    # Plot at most N sims per geometry
    python 02visualizeTrainingData.py --max 10
"""

import argparse
import json
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

METADATA_PATH = 'trainingData/metadata.json'
PML_SIZE = 20   # must match the value used during simulation


def load_sim(h5_path: str) -> dict:
    with h5py.File(h5_path, 'r') as f:
        return {
            'p_rms':     np.array(f['p_rms']),
            'intensity': np.array(f['intensity']),
            'Nx':        int(np.asarray(f.attrs['Nx']).item()),
            'Ny':        int(np.asarray(f.attrs['Ny']).item()),
            'dx':        float(np.asarray(f.attrs['dx_m']).item()),
            'geo':       str(np.asarray(f.attrs['geometry']).item()),
            'n_tx':      int(np.asarray(f.attrs['n_transducers']).item()),
            'x_norms':   np.array(f['transducer_x_norm']),
        }


def interior_slice(arr, Nx: int, Ny: int):
    """Trim PML border to show only the acoustic domain interior."""
    return arr[PML_SIZE:Nx - PML_SIZE, PML_SIZE:Ny - PML_SIZE]


def transducer_x_mm(x_norm: float, Nx: int, dx: float) -> float:
    """
    Convert normalised transducer position to interior-domain mm coordinate.
    The full-grid index is x_norm * Nx; subtract PML_SIZE to get interior index.
    """
    return (x_norm * Nx - PML_SIZE) * dx * 1e3


def plot_sim(meta: dict, save_png: bool = True):
    sim = load_sim(meta['file'])

    p_rms     = interior_slice(sim['p_rms'],     sim['Nx'], sim['Ny'])
    intensity = interior_slice(sim['intensity'], sim['Nx'], sim['Ny'])
    iNx, iNy  = p_rms.shape

    x_mm   = np.arange(iNx) * sim['dx'] * 1e3
    y_mm   = np.arange(iNy) * sim['dx'] * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Geometry: {sim['geo'].upper()}  |  {sim['n_tx']} transducer(s)  "
        f"|  sim {meta['sim_index']:04d}",
        fontsize=13, fontweight='bold'
    )

    def add_transducer_lines(ax):
        for i, xn in enumerate(sim['x_norms']):
            ax.axvline(
                x=transducer_x_mm(xn, sim['Nx'], sim['dx']),
                color='cyan', lw=1.2, ls='--', alpha=0.8,
                label='transducer' if i == 0 else '',
            )
        ax.legend(fontsize=7, loc='lower right')

    # ── Panel 1: RMS pressure ─────────────────────────────────────
    im0 = axes[0].imshow(p_rms.T, origin='upper', aspect='auto',
                         extent=extent, cmap='inferno')
    axes[0].set_title('RMS Pressure (Pa)')
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('depth (mm)  [↓ gravity]')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    add_transducer_lines(axes[0])

    # ── Panel 2: Acoustic intensity ───────────────────────────────
    im1 = axes[1].imshow(intensity.T, origin='upper', aspect='auto',
                         extent=extent, cmap='viridis')
    axes[1].set_title('Acoustic Intensity (W/m²)')
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('depth (mm)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    add_transducer_lines(axes[1])

    # ── Panel 3: RMS pressure depth profile at horizontal centre ──
    cx      = iNx // 2
    profile = p_rms[cx, :]   # pressure along depth at centre x
    mean_p  = np.mean(profile)
    axes[2].plot(profile, y_mm, color='steelblue', lw=1.8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel('RMS Pressure (Pa)')
    axes[2].set_ylabel('depth (mm)')
    axes[2].set_title('Depth profile\n(centre x — gravity uniformity check)')
    axes[2].axvline(x=mean_p, color='tomato', ls='--', lw=1.2,
                    label=f'mean = {mean_p:.2e} Pa')
    axes[2].legend(fontsize=7)

    plt.tight_layout()

    if save_png:
        png = meta['file'].replace('.h5', '_viz.png')
        fig.savefig(png, dpi=150, bbox_inches='tight')
        print(f"  Saved: {png}")

    plt.close(fig)


def select_representative(all_meta: list[dict]) -> list[dict]:
    """
    Pick one simulation per (geometry, n_transducers) pair — the first one
    encountered in metadata order. Used as the default (fast) mode.
    """
    seen = set()
    selected = []
    for m in all_meta:
        key = (m['geometry'], m['n_transducers'])
        if key not in seen:
            seen.add(key)
            selected.append(m)
    return selected


def main():
    parser = argparse.ArgumentParser(description='Visualize k-Wave training data')
    group  = parser.add_mutually_exclusive_group()
    group.add_argument('--all',  action='store_true',
                       help='Plot every simulation (slow for large datasets)')
    group.add_argument('--max',  type=int, metavar='N',
                       help='Plot at most N simulations per geometry')
    args = parser.parse_args()

    with open(METADATA_PATH) as f:
        all_meta = json.load(f)

    print(f"Loaded {len(all_meta)} simulations from {METADATA_PATH}\n")

    if args.all:
        to_plot = all_meta
    elif args.max:
        # Take the first --max entries per geometry
        counts: dict[str, int] = {}
        to_plot = []
        for m in all_meta:
            g = m['geometry']
            counts[g] = counts.get(g, 0)
            if counts[g] < args.max:
                to_plot.append(m)
                counts[g] += 1
    else:
        to_plot = select_representative(all_meta)
        print(f"Plotting {len(to_plot)} representative sims "
              f"(one per geometry × transducer count).\n"
              f"Use --all or --max N to plot more.\n")

    for meta in to_plot:
        print(f"  Plotting sim {meta['sim_index']:04d}  |  "
              f"{meta['geometry']:5s}  |  {meta['n_transducers']} transducer(s)")
        plot_sim(meta, save_png=True)

    print("\nDone. PNG previews saved next to each .h5 file.")


if __name__ == '__main__':
    main()
