from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "sglang" / "sgl-kernel" / "csrc" / "cpu"
INC_DIR = ROOT / "sglang" / "sgl-kernel" / "include"
BUILD_DIR = ROOT / "build" / "decode_ext"
EXT_NAME = "decode_ext"


def find_shared_library() -> Path:
    matches = sorted(BUILD_DIR.glob(f"{EXT_NAME}*.so"))
    if not matches:
        raise FileNotFoundError(f"No shared library matching {EXT_NAME}*.so under {BUILD_DIR}")
    return matches[0]


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    load(
        name=EXT_NAME,
        sources=[str(SRC_DIR / "decode.cpp"), str(ROOT / "decode_binding.cpp")],
        extra_include_paths=[str(SRC_DIR), str(INC_DIR)],
        extra_cflags=["-O3", "-march=native", "-fopenmp", "-Wno-unknown-pragmas"],
        extra_ldflags=["-fopenmp"],
        build_directory=str(BUILD_DIR),
        is_python_module=False,
        verbose=True,
    )
    print(find_shared_library())


if __name__ == "__main__":
    main()