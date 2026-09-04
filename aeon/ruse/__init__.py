"""aeon.ruse — Aeon inside R.U.S.E.: a deception strategy game played on the
layer-by-layer systems map of institutional power.

Pure Python + numpy. The Aeon-backed player (players.AeonPlayer) imports
torch/transformers lazily, so the game, tests, and scripted players run
without a checkpoint.
"""
from .board import Board, Edge, EdgeKind, Faction, Layer, Node, default_board
from .dynamics import DOMAINS, FactionDynamics, default_dynamics
from .engine import Match, MatchResult, RuseConfig
from .latent import Latent
from .observer import Observation, epistemic_loss, observe, render
from .orders import ORDER_HELP, Order, parse_orders
from .players import (AeonPlayer, HeuristicPlayer, LLMPlayer, Player,
                      RandomPlayer, make_player)
from .ruses import Ruse

__all__ = [
    "Board", "Edge", "EdgeKind", "Faction", "Layer", "Node", "default_board",
    "DOMAINS", "FactionDynamics", "default_dynamics",
    "Match", "MatchResult", "RuseConfig", "Latent",
    "Observation", "epistemic_loss", "observe", "render",
    "ORDER_HELP", "Order", "parse_orders",
    "AeonPlayer", "HeuristicPlayer", "LLMPlayer", "Player", "RandomPlayer", "make_player",
    "Ruse",
]
