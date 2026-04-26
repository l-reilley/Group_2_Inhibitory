"""
PING Network Simulation — Izhikevich Model (Final Tuned Version)
=================================================================
NE593 Computational Neuroscience — Group 1

PING (Pyramidal-Interneuron Gamma) mechanism:
  1. RS neurons receive sub-threshold thalamic noise → stochastic firing ~15-25 Hz
  2. RS spikes accumulate AMPA input onto FS interneurons
  3. FS interneurons reach threshold and fire
  4. FS → GABA_A inhibition silences all RS neurons for ~10-15 ms
  5. GABA_A decays; RS rebound and fire together (now synchronized)
  6. Synchronous RS burst strongly recruits FS → clean rhythmic cycle
  Result: ~35-45 Hz gamma oscillation

Key parameter targets:
  - RS thalamic drive must be ABOVE bifurcation (I_bifurc ≈ 4 pA) so neurons fire
    at ~15-25 Hz, providing enough activity to recruit FS neurons
  - w_ie must be strong enough to silence RS but weak enough that GABA_A decays
    within one PING cycle (25 ms for 40 Hz)
  - tau_gabaa ≈ 10 ms is the critical timescale setting the gamma period
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, filtfilt

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ─── Network ───────────────────────────────────────────────────────────────
N_e = 80
N_i = 20
N   = N_e + N_i

# ─── Simulation ────────────────────────────────────────────────────────────
T    = 600.0        # ms
dt   = 0.1          # ms  (single-step Euler, stable for all currents)
n    = int(T / dt)
t_ms = np.arange(n) * dt

# ─── Izhikevich parameters ─────────────────────────────────────────────────
# Heterogeneous excitatory (RS): biases toward RS, some CH
re  = rng.uniform(0, 1, N_e)
a_e = 0.02 * np.ones(N_e)
b_e = 0.20 * np.ones(N_e)
c_e = -65 + 15 * re**2
d_e =   8 -  6 * re**2

# Heterogeneous inhibitory (FS)
ri  = rng.uniform(0, 1, N_i)
a_i = 0.02 + 0.08 * ri
b_i = 0.25 - 0.05 * ri
c_i = -65  * np.ones(N_i)
d_i =   2  * np.ones(N_i)

a = np.concatenate([a_e, a_i])
b = np.concatenate([b_e, b_i])
c = np.concatenate([c_e, c_i])
d = np.concatenate([d_e, d_i])

# ─── Synaptic timescales ────────────────────────────────────────────────────
# These set the key PING timescale: GABA_A decay ≈ half the gamma period.
# With tau_gabaa = 10 ms: GABA decays by 1/e in 10 ms.
# For I_thal = 5.5 and w_ie*14 connections = 10.5 pA of peak inhibition,
# RS neurons rebound to threshold at t ≈ 20-25 ms → f ≈ 35-45 Hz.
tau_ampa  =  5.0   # ms — fast AMPA excitation
tau_gabaa = 10.0   # ms — GABA_A inhibition (KEY parameter for gamma frequency)

# ─── Synaptic weights ──────────────────────────────────────────────────────
# Analytical target for 40 Hz (period 25 ms, rebound at ~20 ms):
#   inhibition at t=20ms: w_ie * n_conn * exp(-20/tau) < I_thal - I_bifurc
#   -> w_ie * 14 * 0.135 < 1.5  ->  w_ie < 0.8  (use 0.75 for margin)
w_ee =  0.05   # E→E  weak recurrent (prevents runaway while allowing sync)
w_ei =  0.70   # E→I  strong (must reliably recruit FS from stochastic RS input)
w_ie = -0.75   # I→E  tuned so GABA silences RS for ~20 ms → gamma cycle
w_ii = -0.05   # I→I  weak self-inhibition

# Connection probabilities
p_ee = 0.15
p_ei = 0.70   # broad E→I connectivity (FS are "listening" to many RS)
p_ie = 0.70   # broad I→E connectivity (FS inhibit the whole RS population)
p_ii = 0.30

def make_conn(n_pre, n_post, p):
    return (rng.uniform(0, 1, (n_post, n_pre)) < p).astype(float)

W_ee = make_conn(N_e, N_e, p_ee)
W_ei = make_conn(N_e, N_i, p_ei)
W_ie = make_conn(N_i, N_e, p_ie)
W_ii = make_conn(N_i, N_i, p_ii)

# ─── Thalamic drive ─────────────────────────────────────────────────────────
# The Izhikevich RS neuron bifurcates at I ≈ 4 (saddle-node).
# Set mean above 4 so neurons fire at ~15-25 Hz without inhibition.
# FS receive NO direct thalamic drive → FS activity is ENTIRELY driven by RS.
I_e_mean = 5.5   # pA  (above bifurcation I≈4 → ~15-25 Hz baseline firing)
I_e_std  = 1.0   # provides neuronal heterogeneity
I_i_mean = 0.0   # FS driven only by RS spikes (pure PING)
I_i_std  = 0.4

# ─── Initial conditions ─────────────────────────────────────────────────────
v = -65 + rng.uniform(-5, 5, N)
u = b * v

s_ampa  = np.zeros(N_e)
s_gabaa = np.zeros(N_i)

# ─── Recording buffers ──────────────────────────────────────────────────────
spikes = []            # (time_ms, neuron_idx)
v_e0   = np.zeros(n)   # one RS neuron — recorded AFTER integration, BEFORE reset
v_i0   = np.zeros(n)   # one FS neuron
lfp    = np.zeros(n)   # mean excitatory v = proxy LFP

# ─── Main loop ──────────────────────────────────────────────────────────────
print(f"PING simulation: {N_e} RS + {N_i} FS | T={T} ms | seed={RNG_SEED}")

for step in range(n - 1):
    # Per-step thalamic noise (different for each neuron and each step)
    I_thal = np.zeros(N)
    I_thal[:N_e] = I_e_mean + I_e_std * rng.standard_normal(N_e)
    I_thal[N_e:] = I_i_mean + I_i_std * rng.standard_normal(N_i)

    # Synaptic currents (current-based with gating variables)
    syn_e = w_ee * (W_ee @ s_ampa) + w_ie * (W_ie @ s_gabaa)  # to RS
    syn_i = w_ei * (W_ei @ s_ampa) + w_ii * (W_ii @ s_gabaa)  # to FS

    I_total = I_thal.copy()
    I_total[:N_e] += syn_e
    I_total[N_e:] += syn_i

    # Single-step Euler (dt=0.1 ms is small enough for stability)
    dv = 0.04 * v**2 + 5 * v + 140 - u + I_total
    du = a * (b * v - u)
    v  = v + dt * dv
    u  = u + dt * du

    # Record AFTER integration, BEFORE reset (captures spike peaks at +30 mV)
    v_e0[step] = v[0]
    v_i0[step] = v[N_e]
    lfp[step]  = v[:N_e].mean()

    # Spike detection and reset
    fired = np.where(v >= 30)[0]
    if fired.size:
        for idx in fired:
            spikes.append((t_ms[step], int(idx)))

        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]

        e_fired = fired[fired < N_e]
        i_fired = fired[fired >= N_e] - N_e
        if e_fired.size:
            s_ampa[e_fired] += 1.0
        if i_fired.size:
            s_gabaa[i_fired] += 1.0

    # Exponential decay of synaptic gating
    s_ampa  *= np.exp(-dt / tau_ampa)
    s_gabaa *= np.exp(-dt / tau_gabaa)

v_e0[-1] = v[0]
v_i0[-1] = v[N_e]
lfp[-1]  = v[:N_e].mean()

print(f"Done. Total spikes: {len(spikes)}")

# ─── Unpack spike data ───────────────────────────────────────────────────────
if spikes:
    spk_t = np.array([s[0] for s in spikes])
    spk_n = np.array([s[1] for s in spikes])
else:
    spk_t, spk_n = np.array([]), np.array([])

e_mask = spk_n < N_e
i_mask = spk_n >= N_e

# ─── Firing rates (10 ms bins) ───────────────────────────────────────────────
bin_ms = 10.0
bins   = np.arange(0, T + bin_ms, bin_ms)
bc     = 0.5 * (bins[:-1] + bins[1:])
er, _  = np.histogram(spk_t[e_mask], bins=bins)
ir, _  = np.histogram(spk_t[i_mask], bins=bins)
er     = er / (N_e * bin_ms * 1e-3)
ir     = ir / (N_i * bin_ms * 1e-3)

# ─── LFP power spectrum (steady-state portion only) ─────────────────────────
skip_ms  = 150.0
skip_i   = int(skip_ms / dt)
lfp_ss   = lfp[skip_i:]
lfp_ss  -= lfp_ss.mean()
fs_hz    = 1000.0 / dt           # sampling frequency in Hz
freqs    = np.fft.rfftfreq(len(lfp_ss), d=1.0 / fs_hz)
psd      = np.abs(np.fft.rfft(lfp_ss))**2 / len(lfp_ss)
fmask    = freqs <= 120

# ─── Mean firing rates ───────────────────────────────────────────────────────
dur = T * 1e-3  # seconds
e_fr = e_mask.sum() / (N_e * dur)
i_fr = i_mask.sum() / (N_i * dur)
print(f"Mean firing rates: RS = {e_fr:.1f} Hz | FS = {i_fr:.1f} Hz")

# ─── Peak gamma frequency ────────────────────────────────────────────────────
gamma_mask = (freqs >= 20) & (freqs <= 100)
if gamma_mask.any():
    peak_f = freqs[gamma_mask][np.argmax(psd[gamma_mask])]
    print(f"Peak oscillation frequency: {peak_f:.1f} Hz")
else:
    peak_f = None

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         FIGURE                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
DARK   = '#0d1117'
PANEL  = '#161b22'
BORDER = '#21262d'
E_C    = '#58a6ff'   # blue  → excitatory / RS
I_C    = '#ff7b72'   # red   → inhibitory / FS
LFP_C  = '#e3b341'   # gold  → LFP
PSD_C  = '#3fb950'   # green → power spectrum
AX_C   = '#8b949e'   # gray  → axis labels

fig = plt.figure(figsize=(15, 14), facecolor=DARK)
gs  = gridspec.GridSpec(
    5, 2, figure=fig,
    height_ratios=[1.4, 0.9, 0.9, 0.9, 1.3],
    hspace=0.60, wspace=0.38,
    left=0.07, right=0.97, top=0.94, bottom=0.05
)

def style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=AX_C, labelsize=8.5)
    ax.xaxis.label.set_color(AX_C)
    ax.yaxis.label.set_color(AX_C)
    ax.grid(axis='x', color=BORDER, lw=0.4, alpha=0.6)
    if title:
        ax.set_title(title, color='white', fontsize=9.5, fontweight='bold', pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color=AX_C, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=AX_C, fontsize=8.5)

# ── A: Raster (full run) ─────────────────────────────────────────────────────
ax_r = fig.add_subplot(gs[0, :])
style(ax_r, 'A   Spike Raster  —  Blue = RS (Excitatory) | Red = FS (Inhibitory)',
      'Time (ms)', 'Neuron index')
if e_mask.any():
    ax_r.scatter(spk_t[e_mask], spk_n[e_mask],
                 s=0.8, c=E_C, alpha=0.65, linewidths=0, label=f'RS (E)  mean {e_fr:.0f} Hz')
if i_mask.any():
    ax_r.scatter(spk_t[i_mask], spk_n[i_mask],
                 s=2.5, c=I_C, alpha=0.90, linewidths=0, label=f'FS (I)  mean {i_fr:.0f} Hz')
ax_r.axhline(N_e - 0.5, color=BORDER, lw=1.0, ls='--')
ax_r.set_xlim(0, T)
ax_r.set_ylim(-1, N)
ax_r.set_yticks([N_e // 2, N_e + N_i // 2])
ax_r.set_yticklabels(['RS\nneurons', 'FS\nneurons'], color=AX_C, fontsize=7.5)
leg = ax_r.legend(loc='upper right', fontsize=7.5,
                  facecolor='#1c2128', edgecolor=BORDER, labelcolor='white', markerscale=5)
ax_r.text(0.02, 0.88,
          'PING cycle: RS fire → FS recruited → GABA silences RS → GABA decays → RS rebound → repeat',
          transform=ax_r.transAxes, color=LFP_C, fontsize=7.5, style='italic',
          bbox=dict(facecolor='#1c2128', edgecolor=BORDER, boxstyle='round,pad=0.25', alpha=0.8))

# ── B: Population firing rates ────────────────────────────────────────────────
ax_fr = fig.add_subplot(gs[1, :])
style(ax_fr, 'B   Population Firing Rate  (10 ms bins)', 'Time (ms)', 'Rate (spk/s)')
ax_fr.plot(bc, er, color=E_C, lw=1.2, label='RS (E)')
ax_fr.plot(bc, ir, color=I_C, lw=1.2, alpha=0.9, label='FS (I)')
ax_fr.set_xlim(0, T)
ax_fr.legend(fontsize=7.5, facecolor='#1c2128', edgecolor=BORDER, labelcolor='white')
ax_fr.text(0.02, 0.82,
           f'Note: FS rate peaks AFTER RS peaks — FS are driven BY RS (PING)',
           transform=ax_fr.transAxes, color=AX_C, fontsize=7.5, style='italic')

# ── C: LFP proxy (full run) ───────────────────────────────────────────────────
ax_l = fig.add_subplot(gs[2, 0])
style(ax_l, 'C   LFP Proxy  (mean RS membrane potential)', 'Time (ms)', 'Mean v (mV)')
ax_l.plot(t_ms, lfp, color=LFP_C, lw=0.6, alpha=0.85)
ax_l.axvline(skip_ms, color=AX_C, lw=0.8, ls=':', alpha=0.5)
ax_l.text(skip_ms + 5, ax_l.get_ylim()[0] * 0.95,
          'analysis\nwindow →', color=AX_C, fontsize=7)
ax_l.set_xlim(0, T)

# ── D: Power spectrum ────────────────────────────────────────────────────────
ax_p = fig.add_subplot(gs[2, 1])
style(ax_p, 'D   LFP Power Spectrum  (steady-state portion)', 'Frequency (Hz)', 'Power (a.u.)')
ax_p.fill_between(freqs[fmask], psd[fmask], color=PSD_C, alpha=0.18)
ax_p.plot(freqs[fmask], psd[fmask], color=PSD_C, lw=1.2)
ax_p.axvspan(25, 100, color=PSD_C, alpha=0.05)
ax_p.axvline(40, color=PSD_C, lw=0.8, ls='--', alpha=0.5)
ax_p.text(42, ax_p.get_ylim()[1] * 0.88, 'γ = 40 Hz',
          color=PSD_C, fontsize=7.5)
ax_p.set_xlim(0, 120)
if peak_f:
    ax_p.axvline(peak_f, color=LFP_C, lw=1.0, ls=':')
    ax_p.text(peak_f + 2, ax_p.get_ylim()[1] * 0.68,
              f'Peak: {peak_f:.0f} Hz', color=LFP_C, fontsize=8, fontweight='bold')

# ── E: LFP zoomed (steady-state) ────────────────────────────────────────────
z0, z1 = 200, 450
zm = (t_ms >= z0) & (t_ms < z1)
ax_lz = fig.add_subplot(gs[3, :])
style(ax_lz, f'E   LFP Proxy Zoomed  ({z0}–{z1} ms)  — oscillatory structure visible here',
      'Time (ms)', 'Mean v (mV)')
ax_lz.plot(t_ms[zm], lfp[zm], color=LFP_C, lw=0.9, alpha=0.9)
ax_lz.set_xlim(z0, z1)
# Annotate period if we have a peak frequency
if peak_f and peak_f > 5:
    period_ms = 1000.0 / peak_f
    ax_lz.annotate('', xy=(z0 + period_ms, ax_lz.get_ylim()[1] * 0.90),
                   xytext=(z0, ax_lz.get_ylim()[1] * 0.90),
                   arrowprops=dict(arrowstyle='<->', color=LFP_C, lw=1.2))
    ax_lz.text(z0 + period_ms / 2, ax_lz.get_ylim()[1] * 0.95,
               f'≈ {period_ms:.0f} ms\n({peak_f:.0f} Hz)',
               color=LFP_C, fontsize=7.5, ha='center', va='top')

# ── F: Single-neuron voltage traces (zoomed) ────────────────────────────────
vm0, vm1 = 300, 450
vm = (t_ms >= vm0) & (t_ms < vm1)
ax_v = fig.add_subplot(gs[4, :])
style(ax_v, f'F   Single-Neuron Voltage Traces  ({vm0}–{vm1} ms)',
      'Time (ms)', 'v (mV)')
ax_v.plot(t_ms[vm], v_e0[vm], color=E_C, lw=0.85, label='RS neuron #0')
ax_v.plot(t_ms[vm], v_i0[vm], color=I_C, lw=0.85, alpha=0.9, label='FS neuron #0')
ax_v.axhline(30, color='white', lw=0.4, ls=':', alpha=0.2)
ax_v.set_xlim(vm0, vm1)
ax_v.legend(fontsize=7.5, facecolor='#1c2128', edgecolor=BORDER, labelcolor='white')
ax_v.text(0.01, 0.88,
          'Spikes shown at +30 mV (reset to c after each spike)',
          transform=ax_v.transAxes, color=AX_C, fontsize=7.5, style='italic')

# ── Figure title ──────────────────────────────────────────────────────────────
pfstr = f" — Peak: {peak_f:.0f} Hz" if peak_f else ""
fig.text(0.5, 0.977,
         f'PING Network  |  Izhikevich Model  ({N_e} RS + {N_i} FS){pfstr}',
         ha='center', fontsize=13, fontweight='bold', color='white')
fig.text(0.5, 0.964,
         'Pyramidal-Interneuron Gamma  |  I_thal(RS)=5.5 pA  |  '
         f'τ_AMPA={tau_ampa:.0f} ms  |  τ_GABA={tau_gabaa:.0f} ms  |  '
         f'w_ie={w_ie}  |  w_ei={w_ei}',
         ha='center', fontsize=8, color=AX_C)

out = 'ping_network_results.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
plt.show()
print(f"Saved: {out}")
