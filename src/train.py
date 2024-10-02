from typing import Dict, Optional
from dataclasses import replace, asdict
from omegaconf import OmegaConf

from rich.progress import Progress
from rich.table import Table


import jax
from jax import Array
from jax import numpy as jnp

import optax
import flax

from lvd.config import Config
from lvd.dataset import Dataset

from lvd.checkpointer import Checkpointer
# from lvd.trainers.cvae import create_trainer
from lvd.trainers.lvd import create_trainer, LVDState

import wandb


INITIALIZATION_BATCH_SIZE: int = 4


def create_metrics_table(metrics: Dict[str, Array]):
    table = Table(title="Training Metrics")

    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    for name, value in metrics.items():
        table.add_row(name, f"{value:.2g}")

    return table


# def filter_checkpoint_keys(state, checkpoint, checkpoint_mask):
#     def filter_checkpoint(item):
#         key, value = item
#         key = ''.join(map(str, key))[1:]

#         for mask in checkpoint_mask:
#             if key[:len(mask)] == mask:
#                 print("INCLUDE", key)
#                 return True

#         return False

#     def anti_filter_checkpoint(item):
#         return not filter_checkpoint(item)
            
#     flat_state, tree_def = jax.tree_util.tree_flatten_with_path(state)
#     flat_checkpoint, _ = jax.tree_util.tree_flatten_with_path(checkpoint)

#     flat_state = list(filter(anti_filter_checkpoint, flat_state))
#     flat_checkpoint = list(filter(filter_checkpoint, flat_checkpoint))

#     flat_state = [x[1] for x in flat_state]
#     flat_checkpoint = [x[1] for x in flat_checkpoint]

#     return jax.tree_util.tree_unflatten(tree_def, flat_state + flat_checkpoint)


def filter_checkpoint_keys(state, checkpoint, checkpoint_mask, negative_checkpoint_mask):
    def is_in_mask(key):
        key = ''.join(map(str, key))[1:]

        for mask in checkpoint_mask:
            if key[:len(mask)] == mask:
                print("EXCLUDE" if negative_checkpoint_mask else "INCLUDE", key)
                return not negative_checkpoint_mask

        return negative_checkpoint_mask
            
    flat_state, tree_def = jax.tree_util.tree_flatten_with_path(state)
    flat_checkpoint, _ = jax.tree_util.tree_flatten_with_path(checkpoint)

    keys = [x[0] for x in flat_state]
    flat_state = [x[1] for x in flat_state]
    flat_checkpoint = [x[1] for x in flat_checkpoint]
    combined = []

    for key, s, c in zip(keys, flat_state, flat_checkpoint):
        if is_in_mask(key):
            combined.append(c)
        else:
            combined.append(s)

    return jax.tree_util.tree_unflatten(tree_def, combined)


def train(
    config_filepath: str,
    dataset_filepath: str,
    weights: Optional[str] = None,
    checkpoint: Optional[str] = None,
    reset_limits: bool = False
):
    config = Config.load(config_filepath)

    # Load dataset and update config with dimensions from data.
    dataset = Dataset(
        dataset_filepath, 
        weight_file=weights,
        include_squared_mass=config.training.consistency_loss_scale > 0, 
    )

    config.dataset = dataset.config
    Config.display(config)

    # Initialization parameters.
    key = jax.random.PRNGKey(config.training.seed)
    example_batch = next(dataset.single_device_dataloader(batch_size=INITIALIZATION_BATCH_SIZE))

    # Construct the trainer object and initialize weights.
    trainer = create_trainer(config)

    state: LVDState = trainer.initialize(
        key, 
        dataset,
        example_batch
    )

    if checkpoint is not None:
        checkpoint_state: LVDState = Checkpointer.load_checkpoint(state, checkpoint)
        if config.training.checkpoint_mask is None:
            state = checkpoint_state
        else:
            state = filter_checkpoint_keys(state, checkpoint_state, config.training.checkpoint_mask, config.training.negative_checkpoint_mask)

    if reset_limits:
        state.lvd_state.params["gamma_limits"]["gamma_max"] = 0.0 * state.lvd_state.params["gamma_limits"]["gamma_max"] + config.noise_schedule.initial_gamma_max
        state.lvd_state.params["gamma_limits"]["gamma_min"] = 0.0 * state.lvd_state.params["gamma_limits"]["gamma_min"] + config.noise_schedule.initial_gamma_min

    # Replicate state across all devices and update the random keys.
    start_step = int(state.step)
    state = flax.jax_utils.replicate(state)
    state = replace(state, seed=jax.pmap(jax.random.fold_in)(state.seed, jnp.arange(jax.device_count())))
    
    # Checkpointing
    checkpointer = Checkpointer(config, state)

    # Logging
    wandb.init(
        project=config.name, 
        config=OmegaConf.to_container(config),
        group=dataset_filepath.split("/")[-1].replace(".npz", ""),
        name=checkpointer.log_folder.split("/")[-1],
    )
    
    dataloader = dataset.multi_device_dataloader(
        batch_size=config.training.batch_size
    )

    # Train the model.
    with Progress() as progress:
        task = progress.add_task("Training", total=config.training.training_steps)

        for step, batch in enumerate(dataloader, start=start_step):
            state, metrics = trainer.update(state, batch)
            metrics = jax.tree_map(lambda x: x.mean().item(), metrics)
            
            if step % config.training.log_interval == 0:
                progress.console.print(create_metrics_table(metrics))
                wandb.log(metrics, step=step)

            if step % config.training.checkpoint_interval == 0:
                checkpointer.save_latest(state)
                checkpointer.save_best(state, metrics["loss"])
  
            if step % config.training.cosine_steps == 0:
                checkpointer.save_checkpoint(state, step)

            if step >= config.training.training_steps:
                break

            progress.advance(task)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath")
    parser.add_argument("dataset_filepath")
    parser.add_argument("--weights", "-w", default=None, help="Path to weights to load.")
    parser.add_argument("--checkpoint", "-c", default=None, help="Path to checkpoint to load.")
    parser.add_argument("--reset-limits", "-r", action="store_true", help="Reset the limits of the dataset.")
    args = parser.parse_args()

    train(
        args.config_filepath,
        args.dataset_filepath,
        args.weights,
        args.checkpoint,
        args.reset_limits
    )
    