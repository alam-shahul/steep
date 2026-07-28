from pathlib import Path

from omegaconf import OmegaConf

from steep.utils import instantiate_from_config


def test_sketcher_configs_instantiate():
    config_dir = Path("steep/config/sketcher")
    for config_path in sorted(config_dir.glob("*.yaml")):
        cfg = OmegaConf.load(config_path)
        sketcher = instantiate_from_config(cfg)
        assert sketcher is not None, config_path
