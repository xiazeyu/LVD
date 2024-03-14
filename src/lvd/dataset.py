from typing import NamedTuple, Dict, Optional
from dataclasses import replace

import tree
import toolz
import numpy as np
import tensorflow as tf

import jax
from jax import Array
from flax.jax_utils import prefetch_to_device

from lvd.config import DatasetConfig, Config


class Batch(NamedTuple):
    detector_vectors: Array
    detector_event: Array
    detector_mask: Array

    particle_vectors: Array
    particle_event: Array
    particle_mask: Array
    particle_types: Array

    particle_weight: Array


class Statistics(NamedTuple):
    mean: Array
    std: Array


class Dataset:
    def __init__(
            self, 
            filepath: str, 
            split: float = 1.0, 
            include_squared_mass: bool = False,
            weight_file: Optional[str] = None
        ):
        with np.load(filepath) as file:
            num_events, max_num_particles = file["particle_mask"].shape
            if weight_file is None:
                particle_weight = np.ones((num_events, max_num_particles), dtype=np.float32)
            else:
                particle_weight = np.load(weight_file)

            print(particle_weight)
            
            self.data = Batch(
                detector_vectors=file["detector_vectors"],
                detector_event=file["detector_event"],
                detector_mask=file["detector_mask"],

                particle_vectors=file["particle_vectors"],
                particle_event=file["particle_event"],
                particle_mask=file["particle_mask"],
                particle_types=file["particle_types"],

                particle_weight=particle_weight[:num_events]
            )

        num_events = self.data.detector_vectors.shape[0]

        if 0 < split < 1:
            split = int(round(num_events * split))
            self.data = tree.map_structure(lambda x: x[:split], self.data)
        elif -1 < split < 0:
            split = int(round(num_events * (split + 1)))
            self.data = tree.map_structure(lambda x: x[split:], self.data)
        elif split == 0 or abs(split) > 1:
            raise ValueError(f"Invalid Split Value: {split}. Split must be a real value in (-1, 1)")

        self.num_events = self.data.detector_vectors.shape[0]
        self.num_particle_types = int(self.particle_types.max() + 1)
        self.dimensions = tree.map_structure(lambda x: x.shape[-1], self.data)

        self.include_squared_mass = include_squared_mass

    @property
    def detector_vectors(self) -> Array:
        return self.data.detector_vectors

    @property
    def detector_event(self) -> Array:
        return self.data.detector_event

    @property
    def detector_mask(self) -> Array:
        return self.data.detector_mask

    @property
    def particle_vectors(self) -> Array:
        return self.data.particle_vectors

    @property
    def particle_event(self) -> Array:
        return self.data.particle_event

    @property
    def particle_mask(self) -> Array:
        return self.data.particle_mask

    @property
    def particle_types(self) -> Array:
        return self.data.particle_types

    @property
    def masked_detector_vectors(self) -> Array:
        return self.detector_vectors[self.detector_mask]

    @property
    def masked_particle_vectors(self) -> Array:
        return self.particle_vectors[self.particle_mask]

    def compute_statistics(self, array: Array) -> Statistics:
        mean = array.mean(0)
        std = array.std(0)
        std = np.where(std < 1e-6, 1.0, std)

        return Statistics(mean, std)

    @property
    def particle_square_mass_statistics(self):
        masked_log_mass = self.particle_vectors[:, :, -1][self.particle_mask]
        masked_mass = np.expm1(masked_log_mass)
        masked_square_mass = masked_mass ** 2

        mean = masked_square_mass.mean()
        std = masked_square_mass.std()

        return mean, std

    @property
    def statistics(self) -> Dict[str, Array]:
        

        variables = [
            ("detector_vector", self.masked_detector_vectors),
            ("detector_event", self.detector_event),

            ("particle_vector", self.masked_particle_vectors),
            ("particle_event", self.particle_event),
            
        ]

        if self.include_squared_mass:
            masked_log_mass = self.masked_detector_vectors[..., -1]
            masked_mass = np.expm1(masked_log_mass)
            masked_square_mass = masked_mass ** 2

            variables.append(("squared_mass", masked_square_mass))
        else:
            variables.append(("squared_mass", np.zeros(self.num_events)))

        names, statistics = zip(*variables)

        statistics = map(self.compute_statistics, statistics)
        statistics = toolz.interleave(zip(*statistics))

        mean_names = [f"{name}_mean" for name in names]
        std_names = [f"{name}_std" for name in names]
        names = toolz.interleave((mean_names, std_names))

        return dict(zip(names, statistics))

    @property
    def config(self) -> DatasetConfig:
        return DatasetConfig(
            detector_vector_dim=self.dimensions.detector_vectors,
            detector_event_dim=self.dimensions.detector_event,

            particle_vector_dim=self.dimensions.particle_vectors,
            particle_event_dim=self.dimensions.particle_event,

            num_particle_types=self.num_particle_types
        )
    
    def update_config(self, config: Config):
        return replace(config, dataset=self.config)

    def multi_device_dataloader(
        self,
        batch_size: int = 1024,
        num_devices: int = None,
        num_epochs: int = None,
        shuffle: bool = True
    ):

        if num_devices is None:
            num_devices = jax.local_device_count()

        devices = jax.local_devices()[:num_devices]

        def split_batch(x):
            return tf.reshape(x, (num_devices, batch_size, *x.shape[1:]))

        def split_batches(*x):
            return tuple(map(split_batch, x))

        with tf.device("CPU"):
            dataset = tf.data.Dataset.from_tensor_slices(tuple(self.data))

            if shuffle:
                dataset = dataset.shuffle(2 * batch_size * num_devices)

            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size * num_devices, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(split_batches, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(Batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return prefetch_to_device(dataset.as_numpy_iterator(), 2, devices=devices)

    def single_device_dataloader(
        self,
        batch_size: int = 1024,
        num_devices: int = None,
    ):

        if num_devices is None:
            num_devices = jax.local_device_count()

        with tf.device("CPU"):
            dataset = tf.data.Dataset.from_tensor_slices(tuple(self.data))

            dataset = dataset.batch(batch_size * num_devices, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(Batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset.as_numpy_iterator()
