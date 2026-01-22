
from itertools import product
import torch

def mnist_circuit(sequence_len=2, n_digits=10, oputput_dim=19):
    possible_worlds = list(product(range(n_digits), repeat=sequence_len))
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, oputput_dim))
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        digit1, digit2 = look_up[w]
        for q in range(n_queries):
            if digit1 + digit2 == q:
                w_q[w, q] = 1
    return w_q

def mnist_sump_circuit(sequence_len=2, n_digits=10, oputput_dim=2):
    possible_worlds = list(product(range(n_digits), repeat=sequence_len))
    n_worlds = len(possible_worlds)
    n_queries = len(range(0, oputput_dim))
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}
    w_q = torch.zeros(n_worlds, n_queries)  # (100, 2)
    for w in range(n_worlds):
        digit1, digit2 = look_up[w]
        for q in range(n_queries):
            if (digit1 + digit2) % 2 == q:
                w_q[w, q] = 1
    return w_q


def mnist_logic(n, entangled):
    if entangled:
        raise NotImplementedError("Entangled mnist logic not implemented.")

    def sum_n_args(input):
        return torch.sum(input, dim=-1)

    return sum_n_args
    