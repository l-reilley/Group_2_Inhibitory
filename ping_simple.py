"""
Simple PING Network — Izhikevich Neurons
=========================================
Two simulations, two raster plots:
  1. RS + FS  → fast GABA decay (tau=8 ms)  → tighter, faster rhythm
  2. RS + LTS → slow GABA decay (tau=20 ms) → broader, slower rhythm

The PING mechanism:
  RS neurons fire → drive inhibitory neurons via AMPA
  Inhibitory neurons fire → suppress RS via GABA
  GABA decays → RS rebounds → next cycle begins
  The GABA time constant (tau_gaba) sets how long the silence lasts,
  which determines the rhythm frequency.
"""

# ─── 1. Import libraries ──────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt


# ─── 2. Izhikevich parameters ─────────────────────────────────────────────────
#
# Equations (each time step dt):
#   dv/dt = 0.04v² + 5v + 140 - u + I
#   du/dt = a(bv - u)
#   if v >= 30 → spike, reset: v = c,  u = u + d

RS  = dict(a=0.02, b=0.20, c=-65, d=8)
FS  = dict(a=0.10, b=0.20, c=-65, d=2, tau_gaba=8)    # fast GABA → shorter silence
LTS = dict(a=0.02, b=0.25, c=-65, d=2, tau_gaba=20)   # slow GABA → longer silence


# ─── 3. Simulation function ───────────────────────────────────────────────────

def run_ping(inhib_params, seed=42):
    """
    Run one PING network and return a list of (time_ms, neuron_index) spikes.

    Synaptic model (simple exponential conductance):
      g_ampa[i]  = AMPA conductance FROM RS neuron i    (shape: N_e)
      g_gaba[i]  = GABA conductance FROM inhibitory i   (shape: N_i)

    Each decays per step: g *= exp(-dt / tau)
    Each spike adds 1.0 to the sender's conductance.
    """
    rng = np.random.default_rng(seed)

    # Network size
    N_e = 80    # RS excitatory   (indices 0-79)
    N_i = 20    # inhibitory      (indices 80-99)

    # Time
    T     = 500.0
    dt    = 0.5
    steps = int(T / dt)

    # ── Izhikevich parameters for all 100 neurons ────────────────────────────
    re = rng.uniform(0, 1, N_e)   # small heterogeneity among RS neurons
    a = np.r_[ 0.02*np.ones(N_e),                inhib_params['a']*np.ones(N_i) ]
    b = np.r_[ 0.20*np.ones(N_e),                inhib_params['b']*np.ones(N_i) ]
    c = np.r_[ -65 + 15*re**2,                   inhib_params['c']*np.ones(N_i) ]
    d = np.r_[   8 -  6*re**2,                   inhib_params['d']*np.ones(N_i) ]

    # ── Initial state ─────────────────────────────────────────────────────────
    v = -65 + rng.uniform(-5, 5, N_e + N_i)
    u = b * v

    # ── Connectivity ──────────────────────────────────────────────────────────
    # W_ei[j, i] = 1 if RS neuron i → inhibitory neuron j  (shape: N_i x N_e)
    W_ei = (rng.random((N_i, N_e)) < 0.7).astype(float)

    # W_ie[j, i] = 1 if inhibitory i → RS neuron j         (shape: N_e x N_i)
    W_ie = (rng.random((N_e, N_i)) < 0.7).astype(float)

    # ── Synaptic conductances ─────────────────────────────────────────────────
    g_ampa = np.zeros(N_e)   # one value per RS neuron   (AMPA sender)
    g_gaba = np.zeros(N_i)   # one value per inhib neuron (GABA sender)

    tau_ampa = 5.0
    tau_gaba = inhib_params['tau_gaba']
    decay_ampa = np.exp(-dt / tau_ampa)
    decay_gaba = np.exp(-dt / tau_gaba)

    w_ampa = 0.8    # strength: RS → inhibitory
    w_gaba = 2.0    # strength: inhibitory → RS

    # ── Background input ──────────────────────────────────────────────────────
    I_thal = 6.0    # thalamic drive to RS only

    # ── Spike recorder ────────────────────────────────────────────────────────
    spikes = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for step in range(steps):
        t = step * dt

        # Total synaptic current arriving at each population
        I_to_inhib = w_ampa * (W_ei @ g_ampa)   # AMPA → inhibitory   shape (N_i,)
        I_to_rs    = w_gaba * (W_ie @ g_gaba)   # GABA → RS            shape (N_e,)

        # Full input current: thalamic drive + synaptic + noise
        I_rs    = I_thal - I_to_rs    + rng.standard_normal(N_e) * 1.5
        I_inhib =          I_to_inhib + rng.standard_normal(N_i) * 0.5

        I_all = np.r_[I_rs, I_inhib]

        # Euler integration
        dv = 0.04*v**2 + 5*v + 140 - u + I_all
        du = a * (b*v - u)
        v += dt * dv
        u += dt * du

        # Detect spikes
        fired_e = np.where(v[:N_e] >= 30)[0]   # RS spikes
        fired_i = np.where(v[N_e:] >= 30)[0]   # inhibitory spikes

        # Record, reset, update conductances
        for idx in fired_e:
            spikes.append((t, int(idx)))
            g_ampa[idx] += 1.0          # this RS neuron now releases AMPA
        for idx in fired_i:
            spikes.append((t, int(idx + N_e)))
            g_gaba[idx] += 1.0          # this inhibitory neuron releases GABA

        if fired_e.size > 0:
            v[:N_e][fired_e]  = c[:N_e][fired_e]
            u[:N_e][fired_e] += d[:N_e][fired_e]
        if fired_i.size > 0:
            v[N_e:][fired_i]  = c[N_e:][fired_i]
            u[N_e:][fired_i] += d[N_e:][fired_i]

        # Conductances decay
        g_ampa *= decay_ampa
        g_gaba *= decay_gaba

    return spikes, N_e


# ─── 4. Run RS + FS simulation ────────────────────────────────────────────────
spikes_fs,  N_e = run_ping(FS,  seed=42)
print(f"RS+FS  — {len(spikes_fs)} spikes")

# ─── 5. Run RS + LTS simulation ───────────────────────────────────────────────
spikes_lts, N_e = run_ping(LTS, seed=42)
print(f"RS+LTS — {len(spikes_lts)} spikes")


# ─── 6. Plot the two raster plots ─────────────────────────────────────────────

def plot_raster(ax, spikes, N_e, title):
    """Scatter one dot per spike, blue for RS, red for inhibitory."""
    if not spikes:
        return
    times   = np.array([s[0] for s in spikes])
    neurons = np.array([s[1] for s in spikes])
    is_inh  = neurons >= N_e
    ax.scatter(times[~is_inh], neurons[~is_inh],
               c='steelblue', s=3, linewidths=0, alpha=0.8, label='RS')
    ax.scatter(times[is_inh],  neurons[is_inh],
               c='tomato',    s=6, linewidths=0, alpha=0.9, label='Inhibitory')
    ax.axhline(N_e - 0.5, color='black', lw=0.9, ls='--', alpha=0.35)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Neuron index", fontsize=10)
    ax.set_xlim(0, 500)
    ax.set_ylim(-1, 100)
    ax.legend(fontsize=8, loc='upper right', markerscale=3)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.suptitle("PING Network — Izhikevich Model  (80 RS + 20 inhibitory neurons)",
             fontsize=13, fontweight='bold')

plot_raster(ax1, spikes_fs,
            N_e=N_e,
            title=f"RS + FS   (τ_GABA = {FS['tau_gaba']} ms)  →  short inhibitory window  →  faster rhythm")
plot_raster(ax2, spikes_lts,
            N_e=N_e,
            title=f"RS + LTS  (τ_GABA = {LTS['tau_gaba']} ms)  →  long inhibitory window  →  slower rhythm")

plt.tight_layout()
plt.savefig("ping_simple.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ping_simple.png")
