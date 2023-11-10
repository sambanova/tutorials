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

import torch
import torch.nn as nn

import sambaflow.samba as samba


def save_checkpoint(model: nn.Module, checkpoint_name: str):
    """Transfer model weights from RDU to host and save them in a PyTorch checkpoint.

    Args:
        model (nn.Module): The model instance to save from
        checkpoint_name (str): The name of the checkpoint
    """
    samba.session.to_cpu(model)

    state_dict = model.state_dict()
    # Save each tensor as a torch Tensor rather than a SambaTensor for portability
    for key, val in state_dict.items():
        if isinstance(val, samba.SambaTensor):
            state_dict[key] = val.torch_tensor()
    print(f"Saving Checkpoint to disk at {checkpoint_name}")
    torch.save(state_dict, checkpoint_name)


def load_checkpoint(model: nn.Module, checkpoint_name: str):
    """Load model weights from a checkpoint on Host and transfer them to RDU.

    Args:
        model (nn.Module): The model instance to load to
        checkpoint_name (str): the name of the checkpoint
    """
    print(f"Loading Checkpoint from disk at {checkpoint_name}")
    checkpoint_state_dict = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint_state_dict)
    samba.session.to_rdu(model)
