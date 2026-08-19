"""UltraTensor — streaming compression for very large models.

UltraTensor is the HyperTensor + HyperRetro combination for the 100B+
class: it takes the compression pipeline HyperRetro proved on small
models (dequantize -> low-cost re-quantize -> factored safetensors +
manifest + certificate-style metadata) and makes it streaming so that
models of ANY size can be processed with bounded RAM.

    from ultratensor.stream import dry_run, compress_gguf

    dry_run(Path("model.gguf"), target="uq4")
    compress_gguf(Path("model.gguf"), Path("out"), target="uq4")
"""

from .dequant import DEQUANT, BLOCK_ALIGN, dequantize
from .quant import (
    q4_0_dequantize,
    q4_0_quantize,
    q8_0_quantize,
    uq4_dequantize,
    uq4_quantize,
)
from .stream import ShardedWriter, TensorCompressor, compress_gguf, dry_run

__version__ = "0.1.0"

__all__ = [
    "dequantize", "DEQUANT", "BLOCK_ALIGN",
    "q8_0_quantize", "q4_0_quantize", "q4_0_dequantize",
    "uq4_quantize", "uq4_dequantize",
    "TensorCompressor", "ShardedWriter", "compress_gguf", "dry_run",
    "__version__",
]
