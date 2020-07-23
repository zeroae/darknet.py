from typing import Iterable

import fsspec
from intake.source.base import DataSource, Schema

from darknet.py import ImageClassifier


class DarknetSource(DataSource):
    name = "darknet"
    version = "0.0.1"
    container = "python"
    partition_access = False

    def __init__(
        self, names, net_config, net_weights, names_slice=None, storage_options=None, metadata=None
    ):
        self._names = names
        names_slice = names_slice or (-1,)
        names_slice = names_slice if isinstance(names_slice, Iterable) else (names_slice,)
        self._names_slice = slice(*names_slice)
        self._net_config = net_config
        self._net_weights = net_weights
        self._storage_options = storage_options or {}
        super().__init__(storage_options, metadata)

    def _get_schema(self):
        return Schema(
            datashape=None, dtype=None, shape=(None,), npartitions=0, extra_metadata=self.metadata
        )

    def _get_partition(self, i):
        with fsspec.open(self._names, mode="rt", encoding="utf-8", **self._storage_options) as f:
            names = [line.rstrip() for line in f.readlines()][self._names_slice]
        with fsspec.open(
            self._net_config, mode="rt", encoding="utf-8", **self._storage_options
        ) as net_config:
            with fsspec.open(self._net_weights, mode="rb", **self._storage_options) as net_weights:
                return ImageClassifier(names, net_config.name, net_weights.name)
