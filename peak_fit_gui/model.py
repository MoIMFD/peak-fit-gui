import csv

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import find_peaks


def get_delimiter(filepath: str, bytes: int = 4096) -> str:
    sniffer = csv.Sniffer()
    with open(filepath, "r") as f:
        data = f.read(bytes)
    sniffed_delimiter = sniffer.sniff(data).delimiter
    return sniffed_delimiter


def load_dataframe(filepath, separator=",", row_skip: int = 1):
    df = pd.read_csv(filepath, sep=separator)
    numeric_columns = df.select_dtypes(include=np.number).columns.values
    df = df[numeric_columns]
    return df.iloc[::row_skip, :].reset_index()


class FindPeaksWrapper:
    # parameter of the find_peaks function except x sequence argument
    param_names = find_peaks.__code__.co_varnames[1:]

    def __init__(self, **kwargs):
        self.params = dict()
        for key, value in kwargs.items():
            self._valid_key(key)
            self.params[key] = key

        self.initial_params = self.params.copy()

    def _valid_key(self, key: str):
        assert (
            key in self.param_names
        ), f"Invalid key <{key}>\nValid keys are: {self.param_names}"

    def update_param(self, key, value):
        self._valid_key(key)
        self.params[key] = value

    @property
    def params_height_reversed(self) -> dict:
        param_dict = self.params.copy()
        if "height" in param_dict.keys():
            param_dict["height"] = [
                -value for value in param_dict["height"][::-1]
            ]
        return param_dict

    def get_pos_peaks(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        return find_peaks(data, **self.params)

    def get_neg_peaks(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        return find_peaks(-data, **self.params_height_reversed)

    def print_peaks(self):
        pass
