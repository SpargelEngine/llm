from os import PathLike
from typing import Literal

type StrOrPath = str | PathLike[str]

type CrossEntropyReduction = Literal["mean", "sum"]
