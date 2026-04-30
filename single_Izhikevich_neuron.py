#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Main simulation function
def izhikevich_single_neuron(params, I_fn, T, dt=0.1, v0=-65.0):
    """
    Simulates a single Izhikevich neuron:
        dv/dt = 0.04v^2 + 5v + 140 - u + I
        du/dt = a(bv - u)

        Reset:
        if v >= 30 mV: v <- c, u <- u + d

    Parameters
    ----------
    a, b, c, d : float
        Izhikevich parameters:
        a : (decay) rate constant for recovery variable 'u' - typically 0.2
        b : sensitivity of recovery variable 'u' - typically 0.02
        c : reset value for voltage 'v' after each spike - typically -65mV - caused by fast K+ conductance
        d : reset value of recovery var 'u' after spiking - typically ~2 - caused by slow Na+ and K+ conductance

    I_fn : callable
        Function of time t (ms) returning input current.
    T : float
        Total time in ms.
    dt : float
        Time step in ms.
    v0 : float
        Initial membrane potential in mV.
    """
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]

    n = int(T / dt)
    t = np.arange(n) * dt

    v = np.zeros(n)
    u = np.zeros(n)
    I = np.zeros(n)
    spikes = []

    v[0] = v0
    u[0] = b * v0

    for i in range(n - 1):
        I[i] = I_fn(t[i])

        dv = 0.04 * v[i]**2 + 5 * v[i] + 140 - u[i] + I[i]
        du = a * (b * v[i] - u[i])

        v[i + 1] = v[i] + dt * dv
        u[i + 1] = u[i] + dt * du

        if v[i + 1] >= 30:
            spikes.append(t[i])
            v[i] = 30
            v[i + 1] = c
            u[i + 1] += d

    I[-1] = I_fn(t[-1])

    return t, v, u, I, np.array(spikes)

def step_current(I_Tstart, amp=10):
    return lambda t: amp if I_Tstart <= t else 0.0


# In[2]:


# -----------------------------
# Define neuron types and colors
# -----------------------------

# Parameters for each neuron type
neuron_types = {
    "RS":  {"a": 0.02, "b": 0.20, "c": -65, "d": 8},   # Regular spiking
    "FS":  {"a": 0.10, "b": 0.20, "c": -65, "d": 2},   # Fast spiking
    "LTS": {"a": 0.02, "b": 0.25, "c": -65, "d": 2}    # Low-threshold spiking
}

colors = {
    "RS": "orange",
    "FS": "green",
    "LTS": "blue"
}


# -----------------------------
# Helper functions
# -----------------------------

def mean_firing_rate(spikes, I_Tstart, T):
    """Mean firing rate from stimulus onset to end of simulation."""
    spikes_after_start = spikes[spikes >= I_Tstart]
    duration_seconds = (T - I_Tstart) / 1000
    return len(spikes_after_start) / duration_seconds


def isi_dynamics(spikes, I_Tstart):
    """
    Computes inter-spike interval and instantaneous firing frequency.

    ISI = time between consecutive spikes.
    Instantaneous frequency = 1000 / ISI.
    """

    spikes_after_start = spikes[spikes >= I_Tstart]

    if len(spikes_after_start) < 2:
        return None, None, None

    isi = np.diff(spikes_after_start)
    isi_time = spikes_after_start[1:] - I_Tstart
    inst_freq = 1000 / isi

    return isi_time, isi, inst_freq


# In[10]:


# -----------------------------
# Run simulations and store results
# -----------------------------

# Simulation settings
T = 500
dt = 0.1
I_Tstart = 20
I_amp = 10

I_fn = step_current(I_Tstart, amp=I_amp)


results = {}

for name, params in neuron_types.items():
    t, v, u, I, spikes = izhikevich_single_neuron(
        params,
        I_fn,
        T=T,
        dt=dt
    )

    rate = mean_firing_rate(spikes, I_Tstart, T)

    results[name] = {
        "t": t,
        "v": v,
        "u": u,
        "I": I,
        "spikes": spikes,
        "rate": rate
    }


# In[11]:


# ------------------
# Figures
# ------------------

plt.figure(figsize=(10, 5))

for name, data in results.items():
    plt.plot(
        data["t"],
        data["v"],
        label=f"{name}: {len(data['spikes'])} spikes, {data['rate']:.1f} Hz",
        color=colors[name],
        linewidth=1.2,
        alpha=0.65
    )

plt.axvline(I_Tstart, color="black", linestyle="--", linewidth=1, label="Stimulus onset")

plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential v (mV)")
plt.title("Single-neuron Izhikevich responses to constant input current")
plt.legend(fontsize='small',loc='upper left')
plt.tight_layout()
plt.show()


# In[12]:


plt.figure(figsize=(10, 5))

for name, data in results.items():
    isi_time, isi, inst_freq = isi_dynamics(data["spikes"], I_Tstart)

    if inst_freq is not None:
        plt.plot(
            isi_time,
            inst_freq,
            marker="o",
            label=name,
            color=colors[name],
            linewidth=1.5
        )

plt.xlabel("Time since stimulus onset (ms)")
plt.ylabel("Instantaneous firing frequency (Hz)")
plt.title("Firing frequency dynamics after stimulus onset")
plt.legend()
plt.tight_layout()
plt.show()


# In[13]:


plt.figure(figsize=(10, 5))

for name, data in results.items():
    isi_time, isi, inst_freq = isi_dynamics(data["spikes"], I_Tstart)

    if isi is not None:
        plt.plot(
            isi_time,
            isi,
            marker="o",
            label=name,
            color=colors[name],
            linewidth=1.5
        )

plt.xlabel("Time since stimulus onset (ms)")
plt.ylabel("Inter-spike interval (ms)")
plt.title("Spike timing adaptation after stimulus onset")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




