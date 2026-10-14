"""Inspect MAT variable metadata without loading the hyperspectral cube.

Run from the repository root:
    python inspect_mat.py data/raw/Chikusei.mat
"""
from __future__ import annotations

import argparse
from pathlib import Path


def inspect_mat(file_path, limit=80):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find data file: {path}")
    print(f"FILE {path.resolve()}")
    print(f"SIZE_BYTES {path.stat().st_size}")
    try:
        import h5py
    except ImportError:
        h5py = None

    if h5py is not None and h5py.is_hdf5(path):
        print("FORMAT HDF5 (MATLAB v7.3 or generic HDF5)")
        count = 0
        with h5py.File(path, "r") as handle:
            print(f"ROOT_KEYS {list(handle.keys())}")

            def show(name, obj):
                nonlocal count
                if count >= limit:
                    print(f"... metadata listing limited to {limit} entries")
                    return True
                count += 1
                matlab_class = obj.attrs.get("MATLAB_class", "")
                if isinstance(matlab_class, bytes):
                    matlab_class = matlab_class.decode("utf-8", errors="replace")
                if isinstance(obj, h5py.Dataset):
                    print(f"/{name}: dataset shape={obj.shape} dtype={obj.dtype} MATLAB_class={matlab_class}")
                else:
                    print(f"/{name}: group MATLAB_class={matlab_class}")

            handle.visititems(show)
        return

    from scipy.io import whosmat
    print("FORMAT classic MAT (metadata read via scipy.io.whosmat)")
    try:
        variables = whosmat(path)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot inspect MAT metadata: {type(exc).__name__}: {exc}. "
            "Check whether this is a complete MAT data file, and install "
            "scipy and h5py for classic/v7.3 support."
        ) from exc
    for name, shape, matlab_class in variables[:limit]:
        print(f"{name}: shape={shape} MATLAB_class={matlab_class}")
    if len(variables) > limit:
        print(f"... metadata listing limited to {limit} entries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", default="data/raw/Chikusei.mat")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be >=1")
    inspect_mat(args.file, args.limit)


if __name__ == "__main__":
    main()
