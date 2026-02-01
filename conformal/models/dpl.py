import torch

from conformal.utils.other import outer_product
from conformal.models.operators import (
    mnist_circuit,
    mnist_sump_circuit,
    boia_circuit,
    chx_circuit,
    derma_circuit,
)
from conformal.models.nesy import NeSyModel


def configure_global_arguments(parser):
    """Global arguments for DeepProbLog"""
    pass


class DPL(NeSyModel):
    def __init__(
        self, n_images, encoder, entangled, concept_dim, output_dim, dataset, device
    ):
        super().__init__(
            n_images, encoder, entangled, concept_dim, output_dim, dataset, device
        )
        self.circuit_data = self._build_circuit(
            concept_dim, output_dim, n_images, dataset
        )

        # If BOIA, circuit is factorized
        if dataset == "boia":
            self.FS_w_q, self.L_w_q, self.R_w_q, self.or_four_bits = [
                t.to(self.device) for t in self.circuit_data
            ]
        else:
            self.circuit = self.circuit_data.to(self.device)

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
        elif dataset == "chx":
            return chx_circuit()
        elif dataset == "derma":
            return derma_circuit()
        raise NotImplementedError(f"Circuit for dataset {dataset} not implemented.")

    def inference(self, concepts, eval=False):
        """DPL-specific probabilistic circuit inference."""
        if self.dataset == "boia":
            return self._boia_inference(concepts), None

        # Standard DPL logic
        worlds = (
            outer_product(concepts.squeeze(1))
            if self.dataset in ["chx", "derma"]
            else outer_product(concepts)
        )
        query_prob = torch.matmul(worlds, self.circuit)
        return self._normalize(query_prob), None

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


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for DPL
    dpl_parser = subparsers.add_parser(
        "dpl",
        help="Use DPL as NeSy predictor",
    )
    configure_global_arguments(dpl_parser)
