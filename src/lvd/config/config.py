
from typing import List, Union, Optional

from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING

from rich import print as rprint
from rich.table import Table

from lvd.config.network_config import NetworkConfig
from lvd.config.noise_schedule_config import NoiseScheduleConfig
from lvd.config.training_config import TrainingConfig
from lvd.config.dataset_config import DatasetConfig

@dataclass
class Schema:
    name: str = MISSING
    base: Optional[str] = None
    
    network: NetworkConfig = field(default_factory=NetworkConfig)
    noise_schedule: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

class Config(Schema):
    @staticmethod
    def load(filepath: str) -> Schema:
        schema = OmegaConf.structured(Schema())
        config = OmegaConf.load(filepath)
        if "base" in config:
            base_config = Config.load(config["base"])
            return OmegaConf.merge(base_config, config)
        else:
            return OmegaConf.merge(schema,  config)
        # return cls(
        #     **{
        #         key: type(getattr(default_instance, key))(**config.get(key, value))
        #         for key, value in asdict(default_instance).items()
        #     }
        # )
    
    @staticmethod
    def save(config: Schema, filepath: str):
        OmegaConf.save(
            config,
            filepath
        )

    @staticmethod
    def display(config: Schema):
        main_table = Table(title="Config")
        main_table.add_column("Option")
        main_table.add_column("Value")

        for name, value in config.items():
            if hasattr(value, "items"):
                table = Table(title=f"{name.capitalize()} Config")
                table.add_column("Option")
                table.add_column("Value")

                for k, v in value.items():
                    table.add_row(k, str(v))

                rprint(table)
            else:
                main_table.add_row(name, str(value))

        rprint(main_table)

                