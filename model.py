import numpy as np
import matplotlib.pyplot as plt
import networkx as x
from numba import njit
from tqdm import tqdm
import torch
import scipy

dtype = torch.float32
K = 7.5
n_r = 6
R = 2 / 3 * 10**6
C = 30 * 10**-12
V_r = -70 * 10**-3
theta = -50 * 10**-3
w_e = 95 * 10**-12
p_r = 0.25
tau_rp = 1 * 10**-3
tau_s = 5 * 10**-3
tau_R = 0.1
device = "cuda"

w_in = 50 * 10**-12
N = 10
f = 5
dt = 0.0001
T = 1


def run(N, f, dt, T, w_in):
    time_steps = int(T / dt)
    poisson_I = torch.tensor(
        np.random.poisson(f * dt, (N, time_steps)), device=device, dtype=dtype
    )
    spikes = torch.zeros((N, time_steps), device=device, dtype=dtype)
    U = torch.zeros((N, time_steps, n_r), device=device, dtype=dtype)
    U[:, 0, :] = 1
    V = torch.zeros(N, device=device, dtype=dtype) + V_r
    Sleep = torch.zeros(N, device=device, dtype=dtype)
    Switch = Sleep == 0

    Graph = x.erdos_renyi_graph(N, K / N, seed=0, directed=False)
    A = torch.zeros((N, N), device=device, dtype=dtype)
    B = x.adjacency_matrix(Graph)
    for edge in list(Graph.edges):
        A[edge[0], edge[1]] = +1

    t_span = torch.tile(torch.arange(0, T, dt, device=device, dtype=dtype), (N, 1))

    def get_I_ext(t):
        I_ext = torch.zeros(N, device=device, dtype=dtype)
        step = int(t / dt)
        fired = torch.zeros((N, time_steps), device=device, dtype=dtype)
        fired[:, :step] = poisson_I[:, :step]
        return w_e * torch.sum(
            (torch.exp((-(t - t_span * poisson_I) / tau_s) * fired) * fired), dim=1
        )

    def get_I_in(t, U, spikes):
        step = int(t / dt)
        opening = p_r * (U * spikes.reshape((N, time_steps, 1)))
        eta = torch.rand(opening.shape, device=device, dtype=dtype)

        fired = torch.zeros((N, time_steps), device=device, dtype=dtype)
        fired[:, :step] = spikes[:, :step]
        gate = torch.sum(
            torch.exp((-(t - t_span * spikes) / tau_s) * fired)
            * fired
            * torch.heaviside(
                opening - eta, torch.zeros_like(eta, device=device, dtype=dtype)
            ).sum(2),
            dim=1,
        )
        return w_in * A @ gate

    def V_dot(V, t, U, spikes):
        return -(V - V_r) / (R * C) + get_I_ext(t) / C + get_I_in(t, U, spikes) / C

    def U_dot(t, U):
        # U_dot = torch.zeros((N, 6), device=device, dtype=dtype)
        step = int(t / dt)
        # opening = spikes[:, step].reshape((-1, 1))
        # eta = torch.rand((N, n_r), device=device, dtype=dtype)
        U_dot = (
            1 - U[:, step]
        ) / tau_R  # - ((torch.heaviside(p_r - eta ,torch.zeros_like(eta,device=device)) * opening) * U[:,step])
        return U_dot

    for step in tqdm(range(time_steps - 1)):
        Sleep -= (Sleep > 0) * dt
        Switch = Sleep <= 0

        t = step * dt

        dU = U_dot(t, U) * dt
        dV = V_dot(V, t, U, spikes) * dt

        U[:, step + 1, :] = U[:, step, :] + dU

        V += dV * Switch

        is_spike = V > theta
        V[is_spike] = V_r
        U[is_spike, step + 1] = 0
        spikes[:, step] = is_spike

        Sleep += tau_rp * is_spike

    torch.save(spikes, f"{N}_{f}_{T}_{w_in*10**12}.pt")

    return spikes
