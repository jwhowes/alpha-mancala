import yaml
import calendar
import time

from typing import Self


class Config:
    @classmethod
    def from_yaml(cls, config_path: str) -> Self:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config is None:
            return cls()

        args = {
            k: float(config.get(k, v.default)) if isinstance(v.default, float) else config.get(k, v.default) for k, v in cls.__dataclass_fields__.items()
        }

        return cls(**args)


def timestamp() -> str:
    return calendar.timegm(time.gmtime())
