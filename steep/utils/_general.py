import hashlib
import importlib
import uuid
from pathlib import Path
from pprint import pformat
from typing import Any

from omegaconf import DictConfig, OmegaConf


def get_name(target: str):
    module, obj = target.rsplit(".", 1)
    name = getattr(importlib.import_module(module, package=None), obj)

    return name, module, obj


def instantiate_from_config(
    config: DictConfig,
    *args: tuple[Any],
    _target_key: str = "type",
    _constructor_key: str = "constructor",
    _params_key: str = "args",
    _disable_key: str = "disable",
    _catch_conflict: bool = True,
    return_name: bool = False,
    **extra_kwargs: Any,
):
    if config.get(_disable_key, False):
        return

    # Obtain target object and kwargs
    cls, module, obj = get_name(config[_target_key])
    kwargs = config.get(_params_key, None) or {}

    constructor_name = config.get(_constructor_key, None)
    if constructor_name is not None:
        constructor = getattr(cls, constructor_name)
    else:
        constructor = cls

    if _catch_conflict:
        assert not (set(kwargs) & set(extra_kwargs)), f"kwargs and extra_kwargs conflicted:\n{kwargs=}\n{extra_kwargs=}"
    full_kwargs = {**kwargs, **extra_kwargs}

    # Instantiate object and handle exception during instantiation
    try:
        if return_name:
            return constructor(*args, **full_kwargs), obj

        return constructor(*args, **full_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {constructor!r} with\nargs:\n{pformat(args)}\nkwargs:\n{pformat(full_kwargs)}",
        ) from e


def hash_config(cfg: DictConfig) -> str:
    """Generate hash for a given config."""
    cfg_str = OmegaConf.to_yaml(cfg, sort_keys=True)
    hex_str = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex_str))


def get_cached_paths(cfg: DictConfig, cache_dir: Path, file_name: str, mkdir: bool) -> tuple[Path, Path]:
    """Get cached data and config path given config."""
    hash_str = hash_config(cfg)

    cache_dir = cache_dir / hash_str
    if mkdir:
        cache_dir.mkdir(exist_ok=True, parents=True)

    cached_file_path = cache_dir / file_name

    return cached_file_path


def filter_config(config, keys_to_keep):
    filtered = OmegaConf.create({})
    for key in keys_to_keep:
        # Use OmegaConf.select() to safely access nested keys using dot notation
        # and set them in the new config
        try:
            value = OmegaConf.select(config, key)
            # Create the nested structure in the filtered config
            OmegaConf.update(filtered, key, value)
        except Exception:
            # Handle cases where the key might be missing if necessary
            pass
    return filtered


def generate_minimal_config(cfg, keys=(), hash_vars=()):
    """Extract keys from OmegaConf in new copy.

    Additionally, add `hash_vars` to the config.

    """
    if len(keys) == 0:
        raise ValueError("Config `keys` used for caching cannot be an empty.")
    cfg = filter_config(cfg, keys_to_keep=keys)
    if len(hash_vars) > 0:
        cfg = {**cfg, "hash_vars": hash_vars}

    return cfg


def get_fully_qualified_cache_paths(
    cfg,
    cache_dir: str | Path,
    filename="",
    keys: set | tuple = (),
    hash_vars=(),
    mkdir: bool = True,
):
    """Get fully-resolved path to unique cache directory, given the config keys
    that distinguish the object being cached."""
    keys = set(keys)  # Make unique

    cfg = generate_minimal_config(cfg, keys=keys, hash_vars=hash_vars)

    fully_qualified_file_path = get_cached_paths(
        cfg,
        Path(cache_dir).resolve(),
        filename,
        mkdir=mkdir,
    )

    return fully_qualified_file_path
