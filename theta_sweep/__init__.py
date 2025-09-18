"""Theta sweep modelling package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("theta-sweep-modelling")
except PackageNotFoundError:  # pragma: no cover - local installs without metadata
    __version__ = "0.0.0"

from . import network_models, plotting  # noqa: E402
from .network_models import DCNet, DCParams, GCNet, GCParams  # noqa: E402

__all__ = [
    "__version__",
    "DCNet",
    "DCParams",
    "GCNet",
    "GCParams",
    "network_models",
    "plotting",
]
