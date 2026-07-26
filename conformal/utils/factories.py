import torch.optim as optim
from conformal.models.resnet18 import ResNet18
from conformal.models.lenet import LeNet
from conformal.models.linear import Linear
from conformal.datasets.mnistadd import MNISTAdditionLoader
from conformal.datasets.mnistsump import MNISTSumParityLoader
from conformal.datasets.mnisthalf import MNISTHalfLoader
from conformal.datasets.mnistevenodd import MNISTEvenOddLoader
from conformal.datasets.mnistaddn import MNISTAdditionNLoader
from conformal.datasets.derma import DERMALoader
from conformal.datasets.boia import BOIALoader, BOIARawImageLoader
from conformal.datasets.chx import CHXLoader
from conformal.datasets.rival import RIVAL10Loader
from conformal.datasets.cifar import CIFAR10Loader
from conformal.datasets.cebab import CeBaBLoader
from conformal.models.bert import Bert
from conformal.models.lama import Lama
from conformal.models.mpnet import MPNetSentence
from conformal.models.dpl import DPL
from conformal.models.ltn import LTN
from conformal.models.linear_predictor import LinearPredictor
from conformal.models.dsl import DSL
from conformal.utils.logic import DSLLogic, LinearLayerLogic


_MNIST_DATASETS = ("mnistadd", "mnistsump", "mnisthalf", "mnistevenodd", "mnistaddn")


def _get_input_norm(dataset):
    if dataset in _MNIST_DATASETS:
        return "mnist"
    if dataset == "cifar":
        return "cifar"
    return "imagenet"


class OptimizerFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_optimizer(optimizer_type, model_parameters, lr=1e-3, momentum=0.9):
        if optimizer_type == "adam":
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer_type == "sgd":
            return optim.SGD(model_parameters, lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class NetworkFactory:
    @staticmethod
    def get_network(
        name: str, input_shape=(1, 28, 28), output_dim=10, args=None, n_images=2,
        concept_names=None,
    ):
        if name.lower() == "resnet18":
            return ResNet18(input_shape=input_shape, num_classes=output_dim)
        elif name.lower() == "lenet":
            return LeNet(input_shape=input_shape, num_classes=output_dim)
        elif name.lower() == "linear":
            return Linear(input_shape=input_shape, num_classes=output_dim)
        elif name.lower() == "bert":
            return Bert(num_concepts=n_images, num_concept_dim=output_dim)
        elif name.lower() == "lama":
            return Lama(num_concepts=n_images, num_concept_dim=output_dim)
        elif name.lower() == "mpnet":
            return MPNetSentence(num_concepts=n_images, num_concept_dim=output_dim)
        elif name.lower() == "clip":
            from conformal.models.clip_encoder import CLIPEncoder
            dataset = getattr(args, "dataset", "")
            texts = CLIPEncoder.build_concept_texts(concept_names, dataset)
            return CLIPEncoder(
                num_classes=output_dim,
                concept_texts=texts,
                model_name=getattr(args, "clip_model", "openai/clip-vit-base-patch32"),
                input_normalization=_get_input_norm(dataset),
            )
        else:
            raise ValueError(f"Unknown network type: {name}")


class DatasetFactory:
    @staticmethod
    def get_dataset(args, name: str, **kwargs):
        name = name.lower()
        if name == "mnistadd":
            return MNISTAdditionLoader(**kwargs).load()
        elif name == "mnistsump":
            return MNISTSumParityLoader(**kwargs).load()
        elif name == "mnisthalf":
            return MNISTHalfLoader(**kwargs).load()
        elif name == "mnistaddn":
            return MNISTAdditionNLoader(n_digits=args.n_digits, **kwargs).load()
        elif name == "mnistevenodd":
            return MNISTEvenOddLoader(**kwargs).load()
        elif name == "boia":
            if getattr(args, "model", "") == "clip":
                return BOIARawImageLoader(
                    raw_root=getattr(args, "boia_raw_root", None), **kwargs
                ).load()
            return BOIALoader(**kwargs).load()
        elif name == "chx":
            return CHXLoader(chx_multi_class=args.chx_multi_class, **kwargs).load()
        elif name == "derma":
            return DERMALoader(**kwargs).load()
        elif name == "cifar":
            return CIFAR10Loader(**kwargs).load()
        elif name == "rival":
            return RIVAL10Loader(**kwargs).load()
        elif name == "cebab":
            if args.model == "bert":
                tokenizer = "bert-base-uncased"
            elif args.model == "lama":
                tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            elif args.model == "mpnet":
                tokenizer = "sentence-transformers/all-mpnet-base-v2"
            else:
                raise ValueError(
                    f"Unknown tokenizer for CeBaB instantiation: {args.model} chosen"
                )
            return CeBaBLoader(tokenizer_name=tokenizer, **kwargs).load()
        else:
            raise ValueError(f"Unknown dataset: {name}")


class NeSyFactory:
    @staticmethod
    def get_nesy_model(
        name: str, n_images, model, concept_dim, output_dim, device, logic, args
    ):
        extra = {"chx-multi-class": getattr(args, "chx_multi_class", False)}

        if name.lower() == "dpl":
            return DPL(
                n_images=n_images,
                encoder=model,
                entangled=args.entangled,
                concept_dim=concept_dim,
                output_dim=output_dim,
                dataset=args.dataset,
                device=device,
                extra=extra,
            )
        elif name.lower() == "ltn":
            return LTN(
                n_images=n_images,
                encoder=model,
                entangled=args.entangled,
                concept_dim=concept_dim,
                output_dim=output_dim,
                dataset=args.dataset,
                device=device,
                logic=logic,
                and_op=args.and_op,
                or_op=args.or_op,
                imp_op=args.imp_op,
                p=args.p,
                extra=extra,
            )
        elif name.lower() in ("linpred", "cbm"):
            return LinearPredictor(
                n_images=n_images,
                encoder=model,
                entangled=args.entangled,
                concept_dim=concept_dim,
                output_dim=output_dim,
                dataset=args.dataset,
                device=device,
            )
        elif name.lower() == "dsl":
            return DSL(
                n_images=n_images,
                encoder=model,
                entangled=args.entangled,
                concept_dim=concept_dim,
                output_dim=output_dim,
                dataset=args.dataset,
                device=device,
                epsilon_rules=args.epsilon_rules,
                epsilon_symbols=args.epsilon_symbols,
            )
        else:
            raise ValueError(f"Unknown nesy method: {name}")


class LogicFactory:
    @staticmethod
    def get_logic(name: str, logic, model):
        if name.lower() in ["dpl", "ltn"]:
            return logic
        elif name.lower() in ("linpred", "cbm"):
            return LinearLayerLogic(
                model=model,
                n_concepts=logic.n_concepts,
                concept_dim=logic.concept_dim,
                is_too_big=logic.is_too_big,
            )
        elif name.lower() == "dsl":
            return DSLLogic(
                model=model,
                n_concepts=logic.n_concepts,
                concept_dim=logic.concept_dim,
                is_too_big=logic.is_too_big,
            )
        else:
            raise ValueError(f"Unknown logic type: {name}")
