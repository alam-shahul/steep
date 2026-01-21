import importlib

def get_name(target: str):
    module, obj = target.rsplit(".", 1)
    name = getattr(importlib.import_module(module, package=None), obj)

    return name, module, obj


def instantiate_from_config(
    config: DictConfig,
    *args: Tuple[Any],
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

    # Instantiate object and handel exception during instantiation
    try:
        if return_name:
            return constructor(*args, **full_kwargs), obj

        return constructor(*args, **full_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {constructor!r} with\nargs:\n{pformat(args)}\nkwargs:\n{pformat(full_kwargs)}",
        ) from e
