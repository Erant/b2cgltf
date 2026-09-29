"""b2cgltf: the only reader and writer of the b2c glTF subject and clip files (SPEC.md)."""
from .document import Document, RuleError, Writer, load  # noqa: F401

__version__ = "1.0.0"
