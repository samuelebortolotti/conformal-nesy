import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from conformal.models import dpl, ltn, linear_predictor, dsl


DATASET_CONCEPT_TEMPLATES = {
    "rival": [
        "an object with wheels",
        "a metallic or shiny object",
        "an object with wings",
        "a living animal",
        "a hairy or furry animal",
        "an animal with horns",
        "an animal with a long snout",
    ],
    "cifar": [
        "an object with wheels",
        "a metallic or shiny object",
        "an object with wings",
        "a living animal",
        "a hairy or furry animal",
        "an animal with horns",
        "an animal with a long snout",
    ],
    "chx": [
        "a chest X-ray showing a bone fracture",
        "a chest X-ray showing pneumothorax",
        "a chest X-ray showing airspace opacity or consolidation",
        "a chest X-ray showing a pulmonary nodule or mass",
    ],
    "derma": [
        "a dermoscopy image of actinic keratoses",
        "a dermoscopy image of basal cell carcinoma",
        "a dermoscopy image of benign keratosis",
        "a dermoscopy image of dermatofibroma",
        "a dermoscopy image of melanocytic nevi",
        "a dermoscopy image of melanoma",
        "a dermoscopy image of vascular lesions",
    ],
    "mnistadd": [
        "a handwritten digit zero",
        "a handwritten digit one",
        "a handwritten digit two",
        "a handwritten digit three",
        "a handwritten digit four",
        "a handwritten digit five",
        "a handwritten digit six",
        "a handwritten digit seven",
        "a handwritten digit eight",
        "a handwritten digit nine",
    ],
    "mnistsump": [
        "a handwritten digit zero",
        "a handwritten digit one",
        "a handwritten digit two",
        "a handwritten digit three",
        "a handwritten digit four",
        "a handwritten digit five",
        "a handwritten digit six",
        "a handwritten digit seven",
        "a handwritten digit eight",
        "a handwritten digit nine",
    ],
    "mnisthalf": [
        "a handwritten digit zero",
        "a handwritten digit one",
        "a handwritten digit two",
        "a handwritten digit three",
        "a handwritten digit four",
    ],
    "mnistevenodd": [
        "a handwritten digit zero",
        "a handwritten digit one",
        "a handwritten digit two",
        "a handwritten digit three",
        "a handwritten digit four",
        "a handwritten digit five",
        "a handwritten digit six",
        "a handwritten digit seven",
        "a handwritten digit eight",
        "a handwritten digit nine",
    ],
    "mnistaddn": [
        "a handwritten digit zero",
        "a handwritten digit one",
        "a handwritten digit two",
        "a handwritten digit three",
        "a handwritten digit four",
        "a handwritten digit five",
        "a handwritten digit six",
        "a handwritten digit seven",
        "a handwritten digit eight",
        "a handwritten digit nine",
    ],
    # BOIA concepts ordered by CONCEPTS_ORDER value (0–20)
    "boia": [
        "a green traffic light",           # 0: green_light
        "a road following direction",       # 1: follow
        "a clear road with no obstacles",   # 2: clear
        "a red traffic light",              # 3: red_light
        "a stop sign",                      # 4: stop_sign
        "a car on the road",                # 5: car
        "a person on the road",             # 6: person
        "a bicycle rider or motorcyclist",  # 7: rider
        "another obstacle on the road",     # 8: other_obstacle
        "no left lane available",           # 9: no_left_lane
        "an obstacle on the left side",     # 10: left_obstacle
        "a solid line on the left",         # 11: left_solid_line
        "a right lane marking",             # 12: right_lane
        "a green traffic light on the right",  # 13: right_green_light
        "a right turn permitted",           # 14: right_follow
        "no right lane available",          # 15: no_right_lane
        "an obstacle on the right side",    # 16: right_obstacle
        "a solid line on the right",        # 17: right_solid_line
        "a left lane marking",              # 18: left_lane
        "a green traffic light on the left",  # 19: left_green_light
        "a left turn permitted",            # 20: left_follow
    ],
}

NORMALIZATION_PRESETS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "cifar":    ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "mnist":    ([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]),
}

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        concept_texts,
        model_name="openai/clip-vit-base-patch32",
        input_normalization="imagenet",
        target_size=224,
    ):
        super().__init__()
        self.target_size = target_size

        self.clip = CLIPModel.from_pretrained(model_name)
        for p in self.clip.parameters():
            p.requires_grad = False

        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        tokens = tokenizer(concept_texts, padding=True, return_tensors="pt")
        with torch.no_grad():
            text_embeds = self.clip.get_text_features(**tokens)
            text_embeds = F.normalize(text_embeds, dim=-1)
        self.register_buffer("text_embeds", text_embeds)

        in_mean, in_std = NORMALIZATION_PRESETS[input_normalization]
        self.register_buffer("in_mean", torch.tensor(in_mean).view(1, 3, 1, 1))
        self.register_buffer("in_std",  torch.tensor(in_std).view(1, 3, 1, 1))
        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std",  torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def _renormalize(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(
                x, size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False,
            )
        x = x * self.in_std + self.in_mean
        return (x - self.clip_mean) / self.clip_std

    def forward(self, x):
        x = self._renormalize(x)
        with torch.no_grad():
            image_embeds = self.clip.get_image_features(pixel_values=x)
        image_embeds = F.normalize(image_embeds, dim=-1)
        logits = image_embeds @ self.text_embeds.T * self.clip.logit_scale.exp()
        return logits

    @staticmethod
    def build_concept_texts(concept_names, dataset):
        if concept_names is not None:
            return [f"a photo of {name}" for name in concept_names]
        templates = DATASET_CONCEPT_TEMPLATES.get(dataset, [])
        if not templates:
            raise ValueError(
                f"No concept templates for dataset '{dataset}'. "
                "Pass concept_names or add an entry to DATASET_CONCEPT_TEMPLATES."
            )
        return templates


def configure_global_arguments(parser):
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model ID for the CLIP backbone.",
    )


def configure_subparsers(subparsers):
    clip_parser = subparsers.add_parser(
        "clip",
        help="Zero-shot CLIP vision-language encoder (fully frozen).",
    )
    configure_global_arguments(clip_parser)

    sub = clip_parser.add_subparsers(dest="nesy")
    ltn.configure_subparsers(sub)
    dpl.configure_subparsers(sub)
    linear_predictor.configure_subparsers(sub)
    dsl.configure_subparsers(sub)
