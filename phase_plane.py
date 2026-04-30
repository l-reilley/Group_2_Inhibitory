"""
Phase Plane Analysis — Izhikevich Neuron Model
===============================================
Run this entire file as one cell in Jupyter, or as a .py script.
It produces three side-by-side phase plane plots: RS, FS, and LTS.
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION: plot one phase plane panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_plane(ax, a, b, I, neuron_name, color_vnull='steelblue', color_unull='tomato'):
    """
    Draw the full phase plane for one Izhikevich neuron type.

    Parameters
    ----------
    ax          : matplotlib axes to draw on
    a           : recovery timescale  (FS: 0.10, RS/LTS: 0.02)
    b           : u-v coupling slope  (LTS: 0.25, RS/FS: 0.20)
    I           : thalamic input current
    neuron_name : string label for the plot title
    """

    # ── STEP 1: build a coarse grid for the vector field ─────────────────────
    # 20×20 grid of (v, u) points covering the biologically relevant region
    v_grid = np.linspace(-80, 30, 20)
    u_grid = np.linspace(-20, 15, 20)
    V, U = np.meshgrid(v_grid, u_grid)   # V and U are both 20×20 arrays

    # ── STEP 2: evaluate derivatives at every grid point ─────────────────────
    DV = 0.04 * V**2 + 5*V + 140 - U + I   # dv/dt at each (v,u)
    DU = a * (b*V - U)                       # du/dt at each (v,u)

    # Normalize so every arrow has the same length (makes the field readable)
    magnitude = np.sqrt(DV**2 + DU**2)
    magnitude[magnitude == 0] = 1            # avoid divide-by-zero
    DV_norm = DV / magnitude
    DU_norm = DU / magnitude

    # ── STEP 3: draw the vector field ────────────────────────────────────────
    ax.quiver(V, U, DV_norm, DU_norm,
              color='darkorange', alpha=0.55,
              scale=28,        # lower number = longer arrows
              width=0.004)

    # ── STEP 4: draw nullclines on a fine grid ────────────────────────────────
    v_fine = np.linspace(-80, 30, 600)

    # v-nullcline: where dv/dt = 0  →  u = 0.04v² + 5v + 140 + I  (parabola)
    u_vnull = 0.04 * v_fine**2 + 5*v_fine + 140 + I

    # u-nullcline: where du/dt = 0  →  u = bv  (straight line)
    u_unull = b * v_fine

    # Only plot the part of the parabola that fits inside the axes window
    visible = (u_vnull >= -20) & (u_vnull <= 15)
    ax.plot(v_fine[visible], u_vnull[visible],
            color=color_vnull, lw=2.5, ls='--', label='v-nullcline  (dv/dt = 0)')
    ax.plot(v_fine, u_unull,
            color=color_unull, lw=2.5, ls='--', label='u-nullcline  (du/dt = 0)')

    # ── STEP 5: find and mark fixed points ───────────────────────────────────
    # Fixed point = intersection of both nullclines.
    # Substituting u = bv into the v-nullcline gives a quadratic in v:
    #   0.04v² + (5−b)v + (140+I) = 0
    A_q = 0.04
    B_q = 5.0 - b
    C_q = 140.0 + I
    disc = B_q**2 - 4 * A_q * C_q

    if disc >= 0:
        # Two real intersections exist
        v_stable   = (-B_q - np.sqrt(disc)) / (2 * A_q)   # lower V → stable rest
        v_unstable = (-B_q + np.sqrt(disc)) / (2 * A_q)   # higher V → threshold

        u_stable   = b * v_stable
        u_unstable = b * v_unstable

        ax.plot(v_stable, u_stable, 'o',
                color='white', ms=11, zorder=6,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'stable FP  (rest, v={v_stable:.0f} mV)')
        ax.plot(v_unstable, u_unstable, 's',
                color='white', ms=10, zorder=6,
                markerfacecolor='none',
                markeredgecolor='black', markeredgewidth=2,
                label=f'unstable FP  (threshold, v={v_unstable:.0f} mV)')

        # Small text annotations next to each point
        ax.annotate(f'rest\n{v_stable:.0f} mV',
                    xy=(v_stable, u_stable),
                    xytext=(v_stable - 10, u_stable + 3.5),
                    fontsize=7.5, color='white',
                    arrowprops=dict(arrowstyle='->', color='white', lw=0.9))
        ax.annotate(f'threshold\n{v_unstable:.0f} mV',
                    xy=(v_unstable, u_unstable),
                    xytext=(v_unstable + 4, u_unstable + 3.5),
                    fontsize=7.5, color='white',
                    arrowprops=dict(arrowstyle='->', color='white', lw=0.9))

    else:
        # Discriminant < 0: no real intersections → neuron cannot rest → must spike
        ax.text(0.5, 0.08,
                f'No fixed points at I={I}\n→ neuron must spike',
                transform=ax.transAxes, ha='center', fontsize=8,
                color='#e3b341', style='italic',
                bbox=dict(facecolor='#1c2128', edgecolor='#444c56',
                          boxstyle='round,pad=0.4'))

    # ── STEP 6: formatting ────────────────────────────────────────────────────
    ax.set_xlim(-80, 30)
    ax.set_ylim(-20, 15)
    ax.set_xlabel('v  (membrane potential, mV)', fontsize=10)
    ax.set_ylabel('u  (recovery variable)', fontsize=10)
    ax.set_title(f'{neuron_name}\nI = {I} pA', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.axvline(0, color='gray', lw=0.5, ls=':')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: create the three-panel figure
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
fig.patch.set_facecolor('#0d1117')

for ax in axes:
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values():
        spine.set_color('#21262d')
    ax.tick_params(colors='#8b949e')
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color('white')

# RS: a=0.02, b=0.20 — I=5.5 puts it above its bifurcation (~4 pA) → no fixed point
plot_phase_plane(axes[0],
                 a=0.02, b=0.20, I=5.5,
                 neuron_name='RS  (a=0.02, b=0.20)',
                 color_vnull='steelblue', color_unull='#58a6ff')

# FS: a=0.10, b=0.20 — same bifurcation as RS, but fast recovery (large a)
plot_phase_plane(axes[1],
                 a=0.10, b=0.20, I=5.5,
                 neuron_name='FS  (a=0.10, b=0.20)',
                 color_vnull='#ff7b72', color_unull='tomato')

# LTS: a=0.02, b=0.25 — bifurcation at ~1 pA, so I=2 already means spiking
plot_phase_plane(axes[2],
                 a=0.02, b=0.25, I=2.0,
                 neuron_name='LTS (a=0.02, b=0.25)',
                 color_vnull='#e3b341', color_unull='goldenrod')

fig.suptitle(
    'Phase Plane Analysis — RS vs FS vs LTS\n'
    'Arrows show trajectory direction at every point. '
    'Nullcline intersections = steady states.',
    fontsize=12, fontweight='bold', color='white', y=1.02
)

plt.tight_layout()
plt.savefig('phase_planes_all.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("Saved: phase_planes_all.png")
