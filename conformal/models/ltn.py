import ltn
from conformal.models.operators import (
    mnist_add_ltn_loss,
    mnist_sump_ltn_loss,
    boia_ltn_loss,
    chx_ltn_loss,
    derma_ltn_loss,
)
from conformal.models.nesy import NeSyModel
import torch
import torch.nn.functional as F


def configure_global_arguments(parser):
    """Configure global arguments for LTN."""
    parser.add_argument(
        "--and_op",
        type=str,
        default="prod",
        help="Semantic for the And Operator",
        choices=["godel", "prod", "luk"],
    )
    parser.add_argument(
        "--or_op",
        type=str,
        default="prod",
        help="Semantic for the Or Operator",
        choices=["godel", "prod", "luk"],
    )
    parser.add_argument(
        "--imp_op",
        type=str,
        default="prod",
        help="Semantic for the Implies Operator",
        choices=["godel", "prod", "luk", "goguen", "klenee"],
    )
    parser.add_argument(
        "--p", type=int, default="7", help="Hyper-parameter for LTN quantifiers grade"
    )


class LTN(NeSyModel):
    def __init__(
        self,
        n_images,
        encoder,
        entangled,
        concept_dim,
        output_dim,
        dataset,
        device,
        logic,
        and_op,
        or_op,
        imp_op,
        p,
        extra,
    ):
        super().__init__(
            n_images, encoder, entangled, concept_dim, output_dim, dataset, device
        )

        self.logic = logic
        self.device = device
        self.and_op = self._build_and(and_op)
        self.or_op = self._build_or(or_op)
        self.imp_op = self._build_imp(imp_op)
        self.exists_op = self._build_exists(p)
        self.forall_op = self._build_forall(p)
        self.not_op = self._build_not()
        self.equiv_op = self._build_equiv(and_op, imp_op)

        self.sat_agg_op = self._build_sat_agg()
        self.ltn_loss = self._get_ltn_loss(concept_dim, output_dim, n_images, dataset, extra)

    def _build_sat_agg(self):
        return ltn.fuzzy_ops.SatAgg()

    def _build_not(self):
        return ltn.Connective(ltn.fuzzy_ops.NotStandard())

    def _build_equiv(self, and_op, imp_op):
        return ltn.Connective(
            ltn.fuzzy_ops.Equiv(
                and_op=self._select_and_operator(and_op),
                implies_op=self._select_imp_operator(imp_op),
            )
        )

    def _build_exists(self, p):
        return ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=p), quantifier="e")

    def _build_forall(self, p):
        return ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f")

    def _select_and_operator(self, and_op):
        if and_op == "godel":
            _and = ltn.fuzzy_ops.AndMin()
        elif and_op == "prod":
            _and = ltn.fuzzy_ops.AndProd()
        else:
            _and = ltn.fuzzy_ops.AndLuk()
        return _and

    def _build_and(self, and_op):
        return ltn.Connective(self._select_and_operator(and_op))

    def _build_or(self, or_op):
        if or_op == "godel":
            _or = ltn.fuzzy_ops.OrMax()
        elif or_op == "prod":
            _or = ltn.fuzzy_ops.OrProbSum()
        else:
            _or = ltn.fuzzy_ops.OrLuk()
        return ltn.Connective(_or)

    def _select_imp_operator(self, imp_op):
        if imp_op == "godel":
            _implies = ltn.fuzzy_ops.ImpliesGodel()
        elif imp_op == "prod":
            _implies = ltn.fuzzy_ops.ImpliesReichenbach()
        elif imp_op == "luk":
            _implies = ltn.fuzzy_ops.ImpliesLuk()
        elif imp_op == "goguen":
            _implies = ltn.fuzzy_ops.ImpliesGoguen()
        else:
            _implies = ltn.fuzzy_ops.ImpliesKleeneDienes()
        return _implies

    def _build_imp(self, imp_op):
        return ltn.Connective(self._select_imp_operator(imp_op))

    def _get_ltn_loss(self, concept_dim, output_dim, n_images, dataset, extra):
        if dataset in ["mnistadd", "mnisthalf", "mnistaddn"]:
            return mnist_add_ltn_loss(
                n_images=n_images,
                and_op=self.and_op,
                exists_op=self.exists_op,
                forall_op=self.forall_op,
                n_outputs=self.output_dim,
            )
        elif dataset == "mnistsump":
            return mnist_sump_ltn_loss(
                and_op=self.and_op,
                exists_op=self.exists_op,
                forall_op=self.forall_op,
                n_outputs=self.output_dim,
            )
        elif dataset == "boia":
            return boia_ltn_loss(
                and_op=self.and_op,
                or_op=self.or_op,
                not_op=self.not_op,
                imp_op=self.imp_op,
                exists_op=self.exists_op,
                forall_op=self.forall_op,
                equiv_op=self.equiv_op,
                sat_agg_op=self.sat_agg_op,
            )
        elif dataset == "chx":
            return chx_ltn_loss(
                equiv_op=self.equiv_op,
                forall_op=self.forall_op,
                not_op=self.not_op,
                exists_op=self.exists_op,
                sat_agg_op=self.sat_agg_op,
                and_op=self.and_op,
                multi_class=extra["chx-multi-class"],
            )
        elif dataset == "derma":
            return derma_ltn_loss(
                equiv_op=self.equiv_op,
                forall_op=self.forall_op,
                not_op=self.not_op,
                exists_op=self.exists_op,
                sat_agg_op=self.sat_agg_op,
            )

        raise NotImplementedError(
            f"LTN SAT-Agg loss for dataset {dataset} not implemented."
        )

    def _inference_boia(self, concepts):
        logic_output = self.logic.forward(concepts)
        block_outputs = []

        for i in range(logic_output.shape[1]):
            one_hot_block = F.one_hot(
                torch.tensor(logic_output[:, i]), num_classes=2
            ).float()
            block_outputs.append(one_hot_block)

        return torch.cat(block_outputs, dim=1).to(self.device)

    def inference(self, concepts, eval=False):
        """Apply the hard logic on the argmax of the concepts"""
        concept_copy = concepts.clone().squeeze().argmax(dim=-1).cpu().numpy()
        if self.dataset == "boia":
            return self._inference_boia(concept_copy), None

        return (
            F.one_hot(
                torch.tensor(self.logic.forward(concept_copy)),
                num_classes=self.output_dim,
            )
            .to(self.device)
            .float(),
            None,
        )

    def compute_loss(
        self,
        dataset,
        criterion,
        conc_pred,
        concepts,
        output,
        target,
        label_weights,
        extra,
    ):
        """Return the LTN loss"""
        return self.ltn_loss(conc_pred, target)


def configure_subparsers(subparsers):
    """Configure subparsers."""
    # Subparser for LTN
    ltn_parser = subparsers.add_parser(
        "ltn",
        help="Use LTN as NeSy predictor",
    )
    configure_global_arguments(ltn_parser)
