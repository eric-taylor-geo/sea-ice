from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

# =========================
# Path containers
# =========================


@dataclass(frozen=True)
class DataPaths:
    x_dir: str
    y_dir: str


@dataclass(frozen=True)
class DatasetSplitPaths:
    # defaults requested
    train_x: str = "../data/train/X"
    train_y: str = "../data/train/Y"
    test_x: str = "../data/test/X"
    test_y: str = "../data/test/Y"


# =========================
# IO: load xarray datasets
# =========================


def load_scene_datasets(
    paths: DataPaths,
    *,
    x_var_primary: str = "nersc_sar_primary",
    x_var_secondary: str = "nersc_sar_secondary",
    y_var: str = "SIC",
    y_suffix: str = "_dmi_prep_reference.nc",
) -> Tuple[Dict[str, xr.Dataset], Dict[str, xr.Dataset]]:
    """
    Load X and Y netCDF datasets into dictionaries keyed by scene_name.

    scene_name is parsed as the substring before the first underscore in X filenames.
    Y filenames are assumed to be: <scene_name> + y_suffix
    """
    x_dict: Dict[str, xr.Dataset] = {}
    y_dict: Dict[str, xr.Dataset] = {}

    if not os.path.isdir(paths.x_dir):
        raise FileNotFoundError(f"X directory not found: {paths.x_dir}")
    if not os.path.isdir(paths.y_dir):
        raise FileNotFoundError(f"Y directory not found: {paths.y_dir}")

    for file_name in os.listdir(paths.x_dir):
        if not file_name.endswith(".nc"):
            continue
        scene_name = file_name.split("_")[0]
        x_path = os.path.join(paths.x_dir, file_name)
        x_dict[scene_name] = xr.open_dataset(x_path)

    for scene_name in list(x_dict.keys()):
        y_path = os.path.join(paths.y_dir, scene_name + y_suffix)
        if os.path.exists(y_path):
            y_dict[scene_name] = xr.open_dataset(y_path)
        else:
            print(
                f"Warning: Y file not found for scene {scene_name}: {y_path}"
            )

    # Fail fast if required variables are missing
    for k, ds in x_dict.items():
        if x_var_primary not in ds or x_var_secondary not in ds:
            raise KeyError(
                f"Scene {k}: expected vars '{x_var_primary}' and '{x_var_secondary}' in X dataset."
            )
    for k, ds in y_dict.items():
        if y_var not in ds:
            raise KeyError(f"Scene {k}: expected var '{y_var}' in Y dataset.")

    return x_dict, y_dict


# =========================
# Core patch sampling
# =========================


def _stack_inputs(
    x_ds: xr.Dataset,
    x_var_primary: str,
    x_var_secondary: str,
) -> np.ndarray:
    """Return (T, X, Y, 2) stacked numpy array from the two SAR channels."""
    a = x_ds[x_var_primary].values
    b = x_ds[x_var_secondary].values
    return np.dstack((a, b))


def patchify_scene(
    x_ds: xr.Dataset,
    y_ds: xr.Dataset,
    *,
    patch_size: int,
    patch_num: int,
    x_var_primary: str = "nersc_sar_primary",
    x_var_secondary: str = "nersc_sar_secondary",
    y_var: str = "SIC",
    nodata_value: int = 255,
    nodata_frac_threshold: float = 0.9,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample random patches from a single scene.

    Returns
    -------
    X : (patch_num, patch_size, patch_size, 2)
    y : (patch_num, patch_size, patch_size)
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if patch_num <= 0:
        raise ValueError("patch_num must be > 0")
    if patch_size % 2 != 0:
        raise ValueError(
            "patch_size must be even because slicing uses +/- patch_size//2."
        )

    rng = random.Random(seed) if seed is not None else random

    X = np.empty((patch_num, patch_size, patch_size, 2), dtype=np.float32)
    y = np.empty((patch_num, patch_size, patch_size), dtype=np.float32)

    data = _stack_inputs(x_ds, x_var_primary, x_var_secondary)
    label = y_ds[y_var].values

    t_max, x_max, _ = data.shape
    half = patch_size // 2

    if t_max <= patch_size or x_max <= patch_size:
        raise ValueError(
            f"Scene too small for patch_size={patch_size}. Got data shape {data.shape}."
        )

    max_tries = (
        10_000  # guard against infinite loops if scene is mostly nodata
    )

    for n in range(patch_num):
        tries = 0
        while True:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    f"Exceeded max_tries={max_tries} while sampling patches. "
                    f"Consider lowering nodata_frac_threshold or patch_size."
                )

            x_n = rng.randint(half, x_max - half)
            t_n = rng.randint(half, t_max - half)

            Xn = data[t_n - half : t_n + half, x_n - half : x_n + half, :]
            yn = label[t_n - half : t_n + half, x_n - half : x_n + half]

            # Reject if too much nodata
            if (
                np.sum(yn == nodata_value)
                <= nodata_frac_threshold * patch_size * patch_size
            ):
                X[n] = Xn
                y[n] = yn
                break

    return X, y


def build_arrays_from_dicts(
    x_dict: Dict[str, xr.Dataset],
    y_dict: Dict[str, xr.Dataset],
    *,
    patch_size: int = 128,
    patches_per_scene: int = 500,
    add_y_channel: bool = True,
    seed: Optional[int] = None,
    x_var_primary: str = "nersc_sar_primary",
    x_var_secondary: str = "nersc_sar_secondary",
    y_var: str = "SIC",
    nodata_value: int = 255,
    nodata_frac_threshold: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Concatenate patches across all scenes that exist in both x_dict and y_dict.

    Returns
    -------
    X : (N, patch_size, patch_size, 2)
    y : (N, patch_size, patch_size) or (N, patch_size, patch_size, 1)
    used_scenes : list[str]
    """
    keys = [k for k in x_dict.keys() if k in y_dict]
    if not keys:
        raise ValueError("No overlapping scenes between X and Y dictionaries.")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    used: List[str] = []

    base_rng = random.Random(seed) if seed is not None else None

    for k in keys:
        scene_seed = (
            base_rng.randint(0, 2**31 - 1) if base_rng is not None else None
        )

        Xp, yp = patchify_scene(
            x_dict[k],
            y_dict[k],
            patch_size=patch_size,
            patch_num=patches_per_scene,
            x_var_primary=x_var_primary,
            x_var_secondary=x_var_secondary,
            y_var=y_var,
            nodata_value=nodata_value,
            nodata_frac_threshold=nodata_frac_threshold,
            seed=scene_seed,
        )

        X_list.append(Xp)
        y_list.append(yp)
        used.append(k)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if add_y_channel:
        y = np.expand_dims(y, axis=-1)

    return X, y, used


# =========================
# Split-level loaders
# =========================


def load_split_arrays(
    x_dir: str,
    y_dir: str,
    *,
    patch_size: int = 128,
    patches_per_scene: int = 500,
    add_y_channel: bool = True,
    seed: Optional[int] = None,
    x_var_primary: str = "nersc_sar_primary",
    x_var_secondary: str = "nersc_sar_secondary",
    y_var: str = "SIC",
    y_suffix: str = "_dmi_prep_reference.nc",
    nodata_value: int = 255,
    nodata_frac_threshold: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one split (train or test) from directories and return concatenated patch arrays.
    """
    paths = DataPaths(x_dir=x_dir, y_dir=y_dir)
    x_dict, y_dict = load_scene_datasets(
        paths,
        x_var_primary=x_var_primary,
        x_var_secondary=x_var_secondary,
        y_var=y_var,
        y_suffix=y_suffix,
    )

    X, y, _ = build_arrays_from_dicts(
        x_dict,
        y_dict,
        patch_size=patch_size,
        patches_per_scene=patches_per_scene,
        add_y_channel=add_y_channel,
        seed=seed,
        x_var_primary=x_var_primary,
        x_var_secondary=x_var_secondary,
        y_var=y_var,
        nodata_value=nodata_value,
        nodata_frac_threshold=nodata_frac_threshold,
    )
    return X, y


def get_train_test_arrays(
    *,
    patch_size: int = 128,
    train_patches_per_scene: int = 500,
    test_patches_per_scene: int = 500,
    add_y_channel: bool = True,
    seed: Optional[int] = None,
    paths: DatasetSplitPaths = DatasetSplitPaths(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: returns (X_train, y_train, X_test, y_test)
    using default split paths:
      train: ../data/train/X, ../data/train/Y
      test:  ../data/test/X,  ../data/test/Y
    """
    X_train, y_train = load_split_arrays(
        paths.train_x,
        paths.train_y,
        patch_size=patch_size,
        patches_per_scene=train_patches_per_scene,
        add_y_channel=add_y_channel,
        seed=seed,
    )

    # different seed so train/test patches aren't identical if directories overlap
    test_seed = None if seed is None else seed + 1

    X_test, y_test = load_split_arrays(
        paths.test_x,
        paths.test_y,
        patch_size=patch_size,
        patches_per_scene=test_patches_per_scene,
        add_y_channel=add_y_channel,
        seed=test_seed,
    )

    return X_train, y_train, X_test, y_test
