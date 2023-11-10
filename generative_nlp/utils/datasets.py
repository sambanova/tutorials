"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

import json
import os
from typing import List, Union

import h5py
import numpy as np
import torch


def is_valid_json(example: str) -> bool:
    """Check if a string is a valid JSON

    Args:
        example (str): The string to check

    Returns:
        bool: Whether the string is a valid JSON or not
    """
    try:
        json.loads(example)
        return True
    except ValueError:
        return False


class GenerativeDataset(torch.utils.data.Dataset):
    """
    A dataset that reads over a file or set of files and produces prompts to generate with.
    """
    def __new__(cls,
                path_or_contents: Union[str, List[str]],
                prompt_key: str = 'prompt',
                completion_key='completion',
                suppress_checks: bool = False):

        path_or_contents = [path_or_contents] if isinstance(path_or_contents, str) else path_or_contents
        dataset = cls._empty_dataset(prompt_key=prompt_key,
                                     completion_key=completion_key,
                                     suppress_checks=suppress_checks)

        for p in path_or_contents:
            if is_valid_json(p):
                dataset += cls.from_json_examples(p)
            elif os.path.isfile(p):
                dataset += cls.from_jsonl_file(p)
            elif os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.endswith('.jsonl'):
                            dataset += cls.from_jsonl_file(root + '/' + f)
            else:
                raise ValueError(f"Provided value '{p}' is neither a .jsonl file nor a json example")

        return dataset

    def __len__(self) -> int:
        """Return the length of the dataset in samples

        Returns:
            int: The length
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get the example at a particular index of the dataset

        Args:
            index (int): The index into the dataset

        Returns:
            torch.Tensor: The example
        """
        return self.examples[idx]

    def __add__(self, other: GenerativeDataset) -> GenerativeDataset:
        """Combine this dataset with another GenerativeDataset

        Args:
            other (GenerativeDataset): The other dataset to add

        Returns:
            GenerativeDataset: The new combined dataset
        """
        new_dataset = GenerativeDataset._empty_dataset(self.prompt_key, self.completion_key, self.suppress_checks)
        new_dataset.examples = self.examples + other.examples
        return new_dataset

    def _check_keys(self) -> None:
        """Checks each example in the dataset to ensure the prompt key is present

        Raises:
            ValueError: Error raised if prompt key is missing from an example
        """
        if self.suppress_checks:
            return
        keys_to_check = [self.prompt_key]
        for e in self.examples:
            for key in keys_to_check:
                if key not in e:
                    raise ValueError(f'JSON key "{key}" not found in example: {e}')

    @classmethod
    def _empty_dataset(cls,
                       prompt_key: str = 'prompt',
                       completion_key: str = 'completion',
                       suppress_checks: bool = False) -> GenerativeDataset:
        """Create an empty GenerativeDataset object with specified prompty & completion keys

        Args:
            prompt_key (str, optional): The key to use for the prompt in each example. Defaults to 'prompt'.
            completion_key (str, optional): The key to use for the completion in each example. Defaults to 'completion'.
            suppress_checks (bool, optional): Whether to suppress the key check or not. Defaults to False.

        Returns:
            GenerativeDataset: The empty dataset
        """
        obj = super().__new__(cls)
        obj.prompt_key = prompt_key
        obj.completion_key = completion_key
        obj.suppress_checks = suppress_checks
        obj.examples = []

        return obj

    @classmethod
    def from_json_examples(cls, examples: Union[str, List[str]]) -> GenerativeDataset:
        """Construct a GenerativeDataset from JSON objects

        Args:
            examples (Union[str, List[str]]): A JSON formatted string or a list of JSON formatted strings

        Returns:
            GenerativeDataset: The constructed GenerativeDataset
        """
        obj = cls._empty_dataset()
        examples = [examples] if isinstance(examples, str) else examples
        obj.examples = [json.loads(e) for e in examples]
        obj._check_keys()

        return obj

    @classmethod
    def from_jsonl_file(cls, paths: str) -> GenerativeDataset:
        """Constructs a GenerativeDataset from a jsonl file

        Args:
            paths (str): Path to the jsonl file

        Raises:
            ValueError: Raise error if provided file path is not a jsonl file

        Returns:
            GenerativeDataset: The constructed dataset
        """
        str_examples = []
        paths = [paths] if isinstance(paths, str) else paths
        for p in paths:
            if not p.endswith('.jsonl'):
                # NOTE: will not work with arrays in top level of jsonl (unexpected case)
                raise ValueError(f"Provided file '{p}' must have .jsonl ext and be in the jsonl format")
            with open(p, 'r') as f:
                lines = f.readlines()
            str_examples += lines

        return cls.from_json_examples(str_examples)


class PretrainingGenerativeDataset(torch.utils.data.Dataset):
    """
    Torch Dataset for a file used in generative tuning
    """
    def __init__(self, input_file):
        self.input_file = input_file
        f = h5py.File(input_file, "r")
        # extra field 'token_type_ids'
        keys = ['input_ids', 'token_type_ids']
        self.inputs = [np.asarray(f[key][:]) for key in keys]

        f.close()

    def __len__(self) -> int:
        """Return the length of the dataset in number of samples

        Returns:
            int: The length
        """
        return len(self.inputs[0])

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        """Get the input IDs and token type IDs at a particular index of the dataset

        Args:
            index (int): The index into the dataset

        Returns:
            List[torch.Tensor]: The input IDs and token type IDs
        """
        [input_ids, token_type_ids] = [torch.from_numpy(input[index].astype(np.int32)) for input in self.inputs]
        return [input_ids.long(), token_type_ids.long()]
