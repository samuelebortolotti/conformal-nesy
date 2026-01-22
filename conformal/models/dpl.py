import torch.nn as nn
import torch

from conformal.utils.other import outer_product
from conformal.models.operators import mnist_circuit, mnist_sump_circuit
from conformal.models.operators import boia_circuit


class DPL(nn.Module):
    def __init__(
        self, n_images, encoder, entangled, concept_dim, output_dim, dataset, device
    ):
        super().__init__()
        self.entangled = entangled
        self.encoder = encoder
        self.n_images = n_images
        self.concept_dim = concept_dim
        self.dataset = dataset
        self.device = device

        # build the circuit
        self.circuit_data = self._build_circuit(
            concept_dim, output_dim, n_images, dataset
        )

        # If BOIA, circuit is factorized
        if dataset == "boia":
            self.FS_w_q, self.L_w_q, self.R_w_q, self.or_four_bits = self.circuit_data
            self.FS_w_q = self.FS_w_q.to(self.device)
            self.L_w_q = self.L_w_q.to(self.device)
            self.R_w_q = self.R_w_q.to(self.device)
            self.or_four_bits = self.or_four_bits.to(self.device)
        else:
            self.circuit = self.circuit_data.to(self.device)

    def _normalize(self, x):
        eps = 1e-5
        x = x + eps
        with torch.no_grad():
            Z = torch.sum(x, dim=-1, keepdim=True)
        x = x / Z
        return x

    def _build_circuit(self, concept_dim, output_dim, n_images, dataset):
        if dataset == "mnistadd" or dataset == "mnisthalf":
            return mnist_circuit(
                sequence_len=n_images, n_digits=concept_dim, oputput_dim=output_dim
            )
        elif dataset == "mnistsump":
            return mnist_sump_circuit(
                sequence_len=n_images, n_digits=concept_dim, oputput_dim=output_dim
            )
        elif dataset == "boia":
            return boia_circuit()
        raise NotImplementedError(f"Circuit for dataset {dataset} not implemented.")

    def get_concepts(self, x):
        if self.dataset in ["cub", "boia"]:
            # Sigmoid for independent binary concepts
            c = torch.sigmoid(self.encoder(x))
            # Expand to [1-p, p] for DPL logic
            c = torch.stack([1 - c, c], dim=-1)
            return self._normalize(c)

        # Softmax for categorical concepts (like MNIST digits)
        c = torch.softmax(self.encoder(x), dim=-1)
        return self._normalize(c)

    def _dpl_inference(self, worlds):
        query_prob = torch.matmul(worlds, self.circuit)  # (B, nr_classes)
        return self._normalize(query_prob)

    def _boia_inference(self, pCs):
        """Factored ProbLog inference for BOIA task"""
        pCs = pCs.squeeze(1)

        # 1. Compute Logic Obstacle (OR of 4 specific bits)
        obs_bits = pCs[:, 10:14]
        # Calculate OR probability: 1 - Prob(all bits are 0)
        prob_all_zero = torch.prod(obs_bits[:, :, 0], dim=1)
        obs_prob = torch.stack([prob_all_zero, 1 - prob_all_zero], dim=1)  # (B, 2)

        # 2. Forward/Stop (FS) Worlds (6 concepts: tl_g, follow, clear, tl_r, t_sign, obs)
        fs_inputs = [pCs[:, 0], pCs[:, 1], pCs[:, 2], pCs[:, 3], pCs[:, 4], obs_prob]
        w_FS = self._compute_worlds(fs_inputs)
        labels_FS = torch.matmul(w_FS, self.FS_w_q)  # (B, 4)

        # 3. Left (L) Worlds
        l_inputs = [pCs[:, 5], pCs[:, 6], pCs[:, 7], pCs[:, 8], pCs[:, 9], pCs[:, 14]]
        w_L = self._compute_worlds(l_inputs)
        labels_L = torch.matmul(w_L, self.L_w_q)  # (B, 2)

        # 4. Right (R) Worlds
        r_inputs = [
            pCs[:, 15],
            pCs[:, 16],
            pCs[:, 17],
            pCs[:, 18],
            pCs[:, 19],
            pCs[:, 20],
        ]
        w_R = self._compute_worlds(r_inputs)
        labels_R = torch.matmul(w_R, self.R_w_q)  # (B, 2)

        # Concatenate all action probabilities
        pred = torch.cat([labels_FS, labels_L, labels_R], dim=1)
        # avoid overflow
        pred = (pred + 1e-5) / (1 + 2 * 1e-5)
        return pred

    def _compute_worlds(self, concept_probs):
        """Computes joint probability of worlds using outer products"""
        res = concept_probs[0]
        for next_c in concept_probs[1:]:
            res = torch.einsum("bi,bj->bij", res, next_c).reshape(res.shape[0], -1)
        return res

    def forward(self, x):
        if self.entangled:
            concepts = self.get_concepts(x)
            y = self._dpl_inference(concepts)

            return y, concepts
        else:

            xs = torch.chunk(x, self.n_images, dim=-1)
            concepts = [self.get_concepts(xi) for xi in xs]
            concepts = torch.stack(concepts, dim=1)

            # MNIST-like datasets
            if self.dataset not in ["boia", "cub"]:
                # compute the possible words
                worlds = outer_product(concepts)
                y = self._dpl_inference(worlds)

            if self.dataset in ["boia"]:
                y_flat = self._boia_inference(concepts)
                y = y_flat.view(y_flat.size(0), -1, 2)

            if self.dataset in ["cub"]:
                raise NotImplementedError("CUB DPL inference not implemented yet.")

            return y, concepts
