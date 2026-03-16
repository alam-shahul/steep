import hydra

from steep.trainer import setup_trainer


@hydra.main(config_path="../steep/config", config_name="config", version_base="1.3")
def main(config):
    trainer = setup_trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
