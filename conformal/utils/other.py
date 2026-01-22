import torch


def outer_product(x):
    B, N, M = x.shape
    vectors = x.unbind(dim=1)

    result = vectors[0]

    for i in range(1, N):
        result = torch.einsum("bi,bj->bij", result, vectors[i])
        result = result.reshape(B, -1)

    return result
