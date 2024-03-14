from lvd.config import Config
from lvd.utils import create_log_folder


import flax
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
from flax.training.train_state import TrainState


class Checkpointer:
    def __init__(self, config: Config, state: TrainState):
        self.config = config

        # Create the destination folder and save a copy of the config.
        self.log_folder = create_log_folder("../checkpoints", config.name)
        Config.save(config, f"{self.log_folder}/config.yaml")

        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.save_args = orbax_utils.save_args_from_target(flax.jax_utils.unreplicate(state))
        self.best_loss = np.inf

    def save_checkpoint(self, state: TrainState, name: str):
        return self.checkpointer.save(
            f'{self.log_folder}/{name}',
            flax.jax_utils.unreplicate(state),
            save_args=self.save_args,
            force=True
        )

    def save_latest(self, state: TrainState):
        return self.save_checkpoint(state, "latest")

    def save_best(self, state: TrainState, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            return self.save_checkpoint(state, "best")

        return False
    
    @classmethod
    def load_checkpoint(cls, state, checkpoint: str):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        return checkpointer.restore(checkpoint, item=state)