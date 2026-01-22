import logging
from typing import Any

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf, open_dict

from train import run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _apply_value(cfg: DictConfig, key: str, value: Any) -> None:
    with open_dict(cfg):
        OmegaConf.update(cfg, key, value, merge=True)


def _apply_fixed_params(cfg: DictConfig) -> None:
    fixed_params = cfg.optuna.get("fixed_params") if "optuna" in cfg else []
    for item in fixed_params or []:
        name = item.get("name")
        value = item.get("value")
        if name is None:
            continue
        _apply_value(cfg, name, value)


def _suggest_params(trial: optuna.Trial, cfg: DictConfig) -> None:
    search_space = cfg.optuna.get("search_space") if "optuna" in cfg else []
    for spec in search_space or []:
        name = spec.get("name")
        if name is None:
            continue
        param_type = str(spec.get("type", "float")).lower()
        if param_type == "float":
            value = trial.suggest_float(
                name,
                float(spec.get("low")),
                float(spec.get("high")),
                log=bool(spec.get("log", False)),
                step=spec.get("step"),
            )
        elif param_type == "int":
            step_val = spec.get("step")
            step = int(step_val) if step_val is not None else 1
            value = trial.suggest_int(
                name,
                int(spec.get("low")),
                int(spec.get("high")),
                log=bool(spec.get("log", False)),
                step=step,
            )
        elif param_type == "categorical":
            choices = spec.get("choices") or []
            if not choices:
                raise ValueError(f"Categorical param {name} must define choices")
            value = trial.suggest_categorical(name, list(choices))
        else:
            raise ValueError(f"Unsupported optuna param type: {param_type}")

        _apply_value(cfg, name, value)


def _create_sampler(cfg: DictConfig) -> optuna.samplers.BaseSampler:
    sampler_name = str(cfg.optuna.get("sampler", "tpe")).lower()
    seed = cfg.optuna.get("seed")
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def _create_pruner(cfg: DictConfig) -> optuna.pruners.BasePruner:
    pruner_name = str(cfg.optuna.get("pruner", "median")).lower()
    if pruner_name == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner()


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    logger.info("Starting training...")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.optuna.get("enabled", False):
        sampler = _create_sampler(cfg)
        pruner = _create_pruner(cfg)
        direction = str(cfg.optuna.get("direction", "minimize")).lower()
        study = optuna.create_study(
            study_name=cfg.optuna.get("study_name"),
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=cfg.optuna.get("storage"),
            load_if_exists=True,
        )

        metric = str(cfg.optuna.get("metric", "fid")).lower()

        def objective(trial: optuna.Trial) -> float:
            trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            _apply_fixed_params(trial_cfg)
            _suggest_params(trial, trial_cfg)

            logger.info("Trial %s config:\n%s", trial.number, OmegaConf.to_yaml(trial_cfg))
            metrics = run(trial_cfg, trial_id=trial.number)
            value = metrics.get("best_fid" if metric == "fid" else "best_is")
            if value is None or value != value:  # NaN check
                return float("inf") if direction == "minimize" else -float("inf")

            trial.set_user_attr("best_fid", metrics.get("best_fid"))
            trial.set_user_attr("best_is", metrics.get("best_is"))
            trial.set_user_attr("avg_fid", metrics.get("avg_fid"))
            trial.set_user_attr("avg_is", metrics.get("avg_is"))
            return float(value)

        study.optimize(objective, n_trials=int(cfg.optuna.get("n_trials", 20)))

        logger.info("Optuna best trial: %s", study.best_trial.number)
        logger.info("Optuna best value: %s", study.best_value)
        logger.info("Optuna best params: %s", study.best_trial.params)
    else:
        run(cfg)

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
