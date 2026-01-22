from itertools import product
import torch

# ===============
# MNIST
# ===============


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


#
# =============== BOIA ===============
#


def boia_circuit():
    FS_w_q = _build_world_queries_matrix_FS()
    L_w_q = _build_world_queries_matrix_L()
    R_w_q = _build_world_queries_matrix_R()
    or_four_bits = _or_four_bits_circuit()

    # return all the circuits
    return FS_w_q, L_w_q, R_w_q, or_four_bits


def _build_world_queries_matrix_FS():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 4
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)  # (100, 20)
    for w in range(n_worlds):
        tl_green, follow, clear, tl_red, t_sign, obs = look_up[w]

        if tl_green + follow + clear > 0:
            if tl_green + tl_red == 2 or clear + obs == 2:
                pass
            elif tl_red + t_sign + obs > 0:
                w_q[w, 0] = 1  # not move
                w_q[w, 3] = 1  # stop
            else:
                w_q[w, 1] = 1  # move forward
                w_q[w, 2] = 1  # no-stop
        else:
            w_q[w, 0] = 1  # not move
            if tl_red + t_sign + obs > 0:
                w_q[w, 3] = 1  # stop
            else:
                w_q[w, 2] = 1  # no-stop
    return w_q


def _build_world_queries_matrix_L():
    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        left_lane, tl_green, follow, no_left_lane, obs, left_solid_line = look_up[w]

        if left_lane + tl_green + follow + no_left_lane + obs + left_solid_line == 0:
            w_q[w, 0] = 0.5
            w_q[w, 1] = 0.5
        elif left_lane + tl_green + follow > 0:
            w_q[w, 1] = 1
        else:
            w_q[w, 0] = 1
    return w_q


def _build_world_queries_matrix_R():

    possible_worlds = list(product(range(2), repeat=6))
    n_worlds = len(possible_worlds)
    n_queries = 2
    look_up = {i: c for i, c in zip(range(n_worlds), possible_worlds)}

    w_q = torch.zeros(n_worlds, n_queries)
    for w in range(n_worlds):
        right_lane, tl_green, follow, no_right_lane, obs, right_solid_line = look_up[w]

        if right_lane + tl_green + follow + no_right_lane + obs + right_solid_line == 0:
            w_q[w, 0] = 0.5
            w_q[w, 1] = 0.5
        elif right_lane + tl_green + follow > 0:
            if obs + right_solid_line + no_right_lane > 0:
                w_q[w, 0] = 1  # not move
            else:
                w_q[w, 1] = 1  # move
        else:
            w_q[w, 0] = 1  # not move
    return w_q


def _or_four_bits_circuit():
    four_bits_or = torch.cat((torch.zeros((16, 1)), torch.ones((16, 1))), dim=1).to(
        dtype=torch.float
    )
    four_bits_or[0] = torch.tensor([1, 0])

    return four_bits_or
