from typing import Optional
from dataclasses import dataclass

from ..util import Config


@dataclass
class MancalaTransformerConfig(Config):
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
