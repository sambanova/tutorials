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

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sambaflow.samba.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration.gpt2_patch import gpt2_patch_helper
from sambaflow import samba
from sambaflow.samba.utils.argparser import parse_app_args
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from utils.checkpoint_utils import load_checkpoint
from utils.datasets import GenerativeDataset


def add_common_args(parser: argparse.ArgumentParser):
    """Adds common arguments to an ArgumentParser object

    Args:
        parser (argparse.ArgumentParser): The argument parser object to add arguments to
    """
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="Path to pretrained model config or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Where to store pretrained models and data downloaded from huggingface.co",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=-1,
        help="The maximum total input sequence length after tokenization. "
        "Data in your data dir will be truncated or padded to this length. ",
    )
    parser.add_argument(
        "--examples_to_generate",
        type=int,
        default=20,
        help="The number of prompts to run generation on",
    )


def add_run_args(parser: argparse.ArgumentParser):
    """Adds arguments used at runtime to an argument parser object

    Args:
        parser (argparse.ArgumentParser): The argument parser object to add arguments to
    """
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to a .json file, .jsonl file or a directory containing .jsonl files. "
        'Each json object should contain a "prompt" key of text used for prompting model text generation.',
    )
    parser.add_argument(
        "--max_tokens_to_generate",
        default=20,
        type=int,
        help="Maximum number of tokens to generate after each prompt.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="",
        help="Path to a checkpoint containing weights with names matching those provided by the --model_name_or_path",
    )


def get_model_trace_inputs(args: argparse.Namespace) -> Tuple[Any]:
    """Get input tensors to use for tracing the model.

    Since they're only used for tracing, these tensors are composed of dummy data.

    Args:
        args (argparse.Namespace): Parsed command line arguments

    Returns:
        Tuple[Any]: Inputs to use for tracing
    """

    batch_size = args.batch_size
    length = args.max_seq_length

    assert batch_size == 1, "Only batch size 1 is supported at the moment"

    # Input IDs
    input_ids = torch.randint(0, 5000, (batch_size, length)).int()
    input_ids = samba.from_torch_tensor(input_ids, name="input_ids")

    # Position IDs
    position_ids = torch.arange(length)
    position_ids = position_ids.short()
    position_ids = samba.from_torch_tensor(
        position_ids.unsqueeze(0).expand(input_ids.shape), name="input_position_ids"
    )

    # Attention Mask
    # Prepare the attention mask for the Hugging Face Module
    attention_mask = torch.randint(2, (batch_size, length), dtype=torch.bfloat16)
    attention_mask = attention_mask[:, None, :].to(torch.float32)
    attention_mask_name = "attention_mask"
    attention_mask = samba.from_torch_tensor(attention_mask, name=attention_mask_name)

    # Items in traced_inputs match the order of inputs to forward() for the model
    traced_inputs = (
        input_ids,
        None,
        attention_mask,
        None,
        position_ids,
        None,
        None,
        None,
    )

    return traced_inputs


def get_runtime_inputs(
    inputs: Dict[str, List[Any]], max_seq_length: int
) -> Sequence[Optional[samba.SambaTensor]]:
    """Given inputs from the dataset, create inputs for samba.session.run.

    These inputs must be the same dtype and shape as the compile inputs

    Args:
        inputs (Dict[str, List[Any]]): Inputs from the data loader
        max_seq_length (int): The max sequence length that the PEF supports

    Returns:
        Sequence[Optional[samba.SambaTensor]]: The named input tensors to use in running the model
    """

    # Create input_ids
    input_ids = inputs["input_ids"]

    # Pad the inputs to the appropriate max sequence length
    input_ids = F.pad(input_ids, (0, max_seq_length - input_ids.shape[1]))
    input_ids = samba.from_torch_tensor(input_ids.int(), name="input_ids")

    # Create attention_mask
    attention_mask = inputs["attention_mask"]
    attention_mask = F.pad(
        attention_mask, (0, max_seq_length - attention_mask.shape[1])
    )
    attention_mask = attention_mask[:, None, :].to(torch.float32)
    attention_mask = samba.from_torch_tensor(attention_mask, name="attention_mask")

    # Create position_ids
    position_ids_torch = torch.arange(max_seq_length).short()
    position_ids = samba.from_torch_tensor(
        position_ids_torch.unsqueeze(0).expand(input_ids.shape),
        name="input_position_ids",
    )

    # Runtime traced inputs match the compile time inputs
    traced_inputs = (
        input_ids,
        None,
        attention_mask,
        None,
        position_ids,
        None,
        None,
        None,
    )

    return traced_inputs


def generate(
    args: argparse.Namespace, model: nn.Module, traced_outputs: Tuple[samba.SambaTensor]
):
    """Generate some outputs from the model, hooking into the Hugging Face generate function.

    Args:
        args (argparse.Namespace): The parsed command line arguments
        model (nn.Module): The transformer model instance
        traced_outputs (Tuple[samba.SambaTensor]): The output tensors generated by the tracing process

    Returns:
        List[str]: A list of predictions from the model
    """

    # Load the checkpoint
    if args.checkpoint_name:
        load_checkpoint(model, args.checkpoint_name)

    # Define the internal forward pass in terms of session.run
    def model_rdu_step(self, *input, **kwargs):
        input_id_length = kwargs["input_ids"].shape[1]
        samba_inputs = get_runtime_inputs(kwargs, args.max_seq_length)

        output_logits = samba.session.run(
            input_tensors=samba_inputs,
            output_tensors=traced_outputs,
            hyperparam_dict={"p": 0.0},
            section_types=["fwd"],
        )[0]
        logits = samba.to_torch(output_logits)[:, :input_id_length, :].float()
        return CausalLMOutputWithCrossAttentions(loss=None, logits=logits)

    # Replace the model's internal forward call with the RDU step call so model_rdu_step is automatically called during generate
    # The Hugging Face model generate function will call the model's forward function to generate text, which will run the
    # model on CPU. To make it run on RDU, we patch the forward function with model_rdu_step
    base_model_class = model.__class__
    base_model_class.__torch_call__ = base_model_class.__call__
    base_model_class.__call__ = model_rdu_step

    # Make a tokenizer. The model checkpoint folder has vocab.json (tokenizer info) and merges.txt files
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )

    # Make a dataset from a .jsonl file or folder of .jsonl files
    dataset = GenerativeDataset(args.data_dir)
    predictions = []

    # Generate predictions from the model
    for k, example in enumerate(dataset):
        if k >= args.examples_to_generate:
            break
        # Tokenize inputs
        model_inputs = tokenizer(example["prompt"], return_tensors="pt")
        input_ids = model_inputs["input_ids"]
        input_length = input_ids.shape[-1]

        # Hook into HF model.generate to generate predictions. The above __call__ patching will ensure the model runs on RDU
        generated_ids = model.generate(
            model_inputs["input_ids"],
            max_length=input_length + args.max_tokens_to_generate,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(generated_ids.squeeze(0))
        predictions.append(generated_text)

    return predictions


def patch_model(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    """Patch the Hugging Face model to make it more efficient when running on RDU.

    Args:
        model (nn.Module): The Hugging Face model instance
        args (argparse.Namespace): The parsed command line args

    Returns:
        nn.Module: The patched model instance
    """
    return gpt2_patch_helper(model)


def main(argv: List[str]) -> None:
    # Parse the args
    args = parse_app_args(
        argv=argv, common_parser_fn=add_common_args, run_parser_fn=add_run_args
    )

    # Download the model from Hugging Face
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_config(config)
    elif args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir
        )
    else:
        raise RuntimeError("Must provide --model_name_or_path or --config_name")

    # Patch the model here
    model = patch_model(model, args)

    samba.from_torch_model_(model)

    inputs = get_model_trace_inputs(args)

    if args.command == "compile":
        samba.session.compile(
            model,
            inputs,
        )
    elif args.command == "run":
        traced_outputs = utils.trace_graph(model, inputs, pef=args.pef)
        predictions = generate(args, model, traced_outputs)
        print(*predictions, sep=f"\n{'-' * 20}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
