from pathlib import Path
import sys

sys.path.append("/home/user/colpali")
import configue
import typer

from colpali_engine.trainer.colmodel_training import (
    ColModelTraining,
    ColModelTrainingConfig,
)
from colpali_engine.utils.gpu_stats import print_gpu_utilization


def main(config_file: Path) -> None:
    print_gpu_utilization()
    print("Loading config")
    config = configue.load(config_file, sub_path="config")
    print("Creating Setup")
    if isinstance(config, ColModelTrainingConfig):
        app = ColModelTraining(config)
    else:
        raise ValueError("Config must be of type ColModelTrainingConfig")

    if config.run_train:
        print("Training model")
        app.train()
        app.save(config_file=config_file)
    if config.run_eval:
        print("Running evaluation")
        app.eval()
    print("Done!")


if __name__ == "__main__":
    main("./scripts/configs/qwen2/train_biqwen2_latent_attn_model.yaml")
