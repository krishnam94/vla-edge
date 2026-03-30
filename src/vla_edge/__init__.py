"""vla-edge: Profile, optimize, validate, and deploy VLA models to edge hardware."""

try:
    from vla_edge._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
