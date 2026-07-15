from pathlib import Path

_decode_module = None
_decode_kernel = None


def get_decode_kernel():
    global _decode_module, _decode_kernel

    if _decode_kernel is None:
        import cupy as cp

        ptx_path = Path(__file__).resolve().with_name("decode.ptx")

        if not ptx_path.exists():
            raise FileNotFoundError(f"DFloat11 decode PTX not found: {ptx_path}")

        _decode_module = cp.RawModule(path=str(ptx_path))
        _decode_kernel = _decode_module.get_function("decode")

    return _decode_kernel