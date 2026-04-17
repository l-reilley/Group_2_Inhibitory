
import numpy as np
import matplotlib.pyplot as plt

def izhikevich_sim(a, b, c, d, I_fn, T=300.0, dt=0.1, v0=-65.0):
    """
    Simulate the Izhikevich neuron model.
    dv/dt = 0.04v^2 + 5v + 140 - u + I
    du/dt = a(bv - u)
    if v >= 30 mV: v <- c, u <- u + d

    Parameters
    ----------
    a, b, c, d : float
        Izhikevich parameters.
    I_fn : callable
        Function of time t (ms) returning input current.
    T : float
        Total time in ms.
    dt : float
        Time step in ms.
    v0 : float
        Initial membrane potential.
    """
    n = int(T / dt)
    t = np.arange(n) * dt
    v = np.zeros(n)
    u = np.zeros(n)
    I = np.zeros(n)

    v[0] = v0
    u[0] = b * v0

    for i in range(n - 1):
        I[i] = I_fn(t[i])

        dv = 0.04 * v[i]**2 + 5 * v[i] + 140 - u[i] + I[i]
        du = a * (b * v[i] - u[i])

        v[i + 1] = v[i] + dt * dv
        u[i + 1] = u[i] + dt * du

        if v[i + 1] >= 30:
            v[i] = 30  # show spike peak in plot
            v[i + 1] = c
            u[i + 1] += d

    I[-1] = I_fn(t[-1])
    return t, v, u, I


def step_current(start=50, stop=250, amp=10):
    return lambda t: amp if start <= t <= stop else 0.0


if __name__ == "__main__":
    # Fast-spiking inhibitory interneuron (FS)
    fs_params = dict(a=0.1, b=0.2, c=-65, d=2)

    # Low-threshold spiking inhibitory interneuron (LTS)
    # Based on the paper: larger b gives lower threshold and adaptation.
    # Common educational parameter set:
    lts_params = dict(a=0.02, b=0.25, c=-65, d=2)

    I_fn = step_current(start=50, stop=250, amp=10)

    t_fs, v_fs, u_fs, I_fs = izhikevich_sim(**fs_params, I_fn=I_fn)
    t_lts, v_lts, u_lts, I_lts = izhikevich_sim(**lts_params, I_fn=I_fn)

    plt.figure(figsize=(10, 6))
    plt.plot(t_fs, v_fs, label="FS inhibitory neuron")
    plt.plot(t_lts, v_lts, label="LTS inhibitory neuron")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential v (mV)")
    plt.title("Izhikevich inhibitory neuron dynamics")
    plt.legend()
    plt.tight_layout()
    plt.show()
