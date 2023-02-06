import torch

from collections import Counter
from alpaca.uncertainty_estimator.masks import build_mask
from typing import Iterable, Union, Dict


import numpy as np
import time
import random

import logging

log = logging.getLogger(__name__)

class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(
            x, self.p, training=self.training or self.activate
        )


def convert_to_mc_dropout(
    model: torch.nn.Module, substitution_dict: Dict[str, torch.nn.Module] = None
):
    for i, layer in enumerate(list(model.children())):
        proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name in substitution_dict.keys():
            model._modules[module_name] = substitution_dict[layer_name](
                p=getattr(layer, proba_field_name), activate=False
            )
        else:
            convert_to_mc_dropout(model=layer, substitution_dict=substitution_dict)


def activate_mc_dropout(
    model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False
):
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            layer.activate = activate
            if activate and random:
                layer.p = random
            if not activate:
                layer.p = layer.p_init
        else:
            activate_mc_dropout(
                model=layer, activate=activate, random=random, verbose=verbose
            )

def convert_dropouts(model, dropout_type='MC'):
    if dropout_type == 'MC':
        dropout_ctor = lambda p, activate: DropoutMC(
            p=0.1, activate=False
          )
    elif dropout_type == "DPP":

      def dropout_ctor(p, activate):
        return DropoutDPP(
          p=p,
          activate=activate,
          max_n=100,
          max_frac=0.8,
          mask_name="dpp",
        )

    else:
      raise ValueError(f"Wrong dropout type: {ue_args.dropout_type}")

    # set_last_dropout(model, dropout_ctor(p=ue_args.inference_prob, activate=False))
    if dropout_type == "DPP":
        model.dropout = dropout_ctor(p=0.1, activate=False)
    else:
        convert_to_mc_dropout(model, {"Dropout": dropout_ctor})
        # model.dropout = DropoutDPP(
        #   p=0.1,
        #   activate=False,
        #   max_n=100,
        #   max_frac=0.8,
        #   mask_name="dpp",
        # )
