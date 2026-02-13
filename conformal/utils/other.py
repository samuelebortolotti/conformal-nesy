import torch
import argparse


def outer_product(x):
    B, N, M = x.shape
    vectors = x.unbind(dim=1)

    result = vectors[0]

    for i in range(1, N):
        result = torch.einsum("bi,bj->bij", result, vectors[i])
        result = result.reshape(B, -1)

    return result

def int_ge_2(value):
    ivalue = int(value)
    if ivalue < 2:
        raise argparse.ArgumentTypeError("p must be >= 2")
    return ivalue