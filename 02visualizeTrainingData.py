# cd /Users/davisbone/Repositories/Bespoke-Ultrasound
# .venv/bin/python visualizeTrainingData.py

"""
Visualization of k-Wave acoustic pressure field training data.

For each geometry (well, slide) plots one figure per transducer count showing:
  - RMS pressure field
  - Radiation force density (the "gravity-like" body force)
  - Vertical profile of radiation force at the horizontal center
    (uniform profile = good gravity analog)

Output PNGs are saved alongside the .h5 files.
"""

import json
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

METADATA_PATH = 'trainingData/metadata.json'
PML_SIZE = 20   # must match the value used during simulation


def load_sim(h5_path: str):
    with h5py.File(h5_path, 'r') as f:
        p_rms    = f['p_rms'][:]
        rf_dens  = f['radiation_force_dens'][:]
        Nx       = int(f.attrs['Nx'])
        Ny       = int(f.attrs['Ny'])
        dx       = float(f.attrs['dx_m'])
        geo      = str(f.attrs['geometry'])
        n_tx     = int(f.attrs['n_transducers'])
        x_norms  = f['transducer_x_norm'][:]
    return p_rms, rf_dens, Nx, Ny, dx, geo, n_tx, x_norms


def interior_slice(arr, Nx, Ny):
    """Trim PML border to show only the acoustic domain interior."""
    return arr[PML_SIZE:Nx - PML_SIZE, PML_SIZE:Ny - PML_SIZE]


def plot_sim(meta: dict, save_png: bool = True):
    p_rms, rf_dens, Nx, Ny, dx, geo, n_tx, x_norms = load_sim(meta['file'])

    # Trim to interior domain (remove PML padding)
    p_rms   = interior_slice(p_rms,   Nx, Ny)
    rf_dens = interior_slice(rf_dens, Nx, Ny)
    iNx, iNy = p_rms.shape

    x_mm = np.arange(iNx) * dx * 1e3
    y_mm = np.arange(iNy) * dx * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        f"Geometry: {geo.upper()}  |  {n_tx} transducer(s)",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # ── Panel 1: RMS pressure ─────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(p_rms.T, origin='upper', aspect='auto',
                     extent=extent, cmap='inferno')
    ax0.set_title('RMS Pressure (Pa)')
    ax0.set_xlabel('x (mm)')
    ax0.set_ylabel('depth (mm)  [↓ gravity]')
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    for xn in x_norms:
        ax0.axvline(x=xn * x_mm[-1], color='cyan', lw=1.2, ls='--', alpha=0.8,
                    label='transducer' if xn == x_norms[0] else '')
    ax0.legend(fontsize=7, loc='lower right')

    # ── Panel 2: Radiation force density ─────────────────────────
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(rf_dens.T, origin='upper', aspect='auto',
                     extent=extent, cmap='viridis')
    ax1.set_title('Radiation Force Density (N/m³)')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('depth (mm)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # ── Panel 3: Vertical profile at horizontal center ────────────
    ax2 = fig.add_subplot(gs[2])
    cx = iNx // 2
    profile = rf_dens[cx, :]          # force along depth at center x
    ax2.plot(profile, y_mm, color='steelblue', lw=1.8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Force density (N/m³)')
    ax2.set_ylabel('depth (mm)')
    ax2.set_title('Vertical profile\n(center x)')
    ax2.axvline(x=np.mean(profile), color='tomato', ls='--', lw=1.2,
                label=f'mean = {np.mean(profile):.2e}')
    ax2.legend(fontsize=7)

    plt.tight_layout()

    if save_png:
        png = meta['file'].replace('.h5', '_viz.png')
        fig.savefig(png, dpi=150, bbox_inches='tight')
        print(f"  Saved: {png}")

    plt.close(fig)


def main():
    with open(METADATA_PATH) as f:
        all_meta = json.load(f)

    print(f"Loaded {len(all_meta)} simulations from {METADATA_PATH}\n")

    for meta in all_meta:
        geo   = meta['geometry']
        n_tx  = meta['n_transducers']
        idx   = meta['sim_index']
        print(f"  Plotting sim {idx:04d}  |  {geo:5s}  |  {n_tx} transducer(s)")
        plot_sim(meta, save_png=True)

    print("\nDone. PNG previews saved next to each .h5 file.")


if __name__ == '__main__':
    main()
