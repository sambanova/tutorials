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
import itertools
import math
import os
import random
import sys
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from configuration.gpt2_patch import gpt2_patch_helper
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from utils.checkpoint_utils import save_checkpoint
from utils.datasets import PretrainingGenerativeDataset
from utils.globals import (COMPLETION_TOKEN_TYPE_ID, PADDING_TOKEN_TYPE_ID,
                           PROMPT_TOKEN_TYPE_ID)

import sambaflow.samba.utils as utils
from sambaflow import samba
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.sambacollections import OrderedSet


def add_common_args(parser: argparse.ArgumentParser):
    """Adds common arguments to an argparser object

    Args:
        parser (argparse.ArgumentParser): The argument parser object to add arguments to
    """
    parser.add_argument('--model_name_or_path',
                        type=str,
                        help='Path to pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--config_name',
                        type=str,
                        help='Path to pretrained model config or model identifier from huggingface.co/models')
    parser.add_argument('--cache_dir',
                        type=str,
                        help='Where to store pretrained models and data downloaded from huggingface.co')
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=-1,
                        help='The maximum total input sequence length after tokenization. '
                        'Data in your data dir will be truncated or padded to this length. ')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.1,
                        help='The weight decay to apply (if not zero) to all layers except all '
                        'bias and LayerNorm weights in the AdamW optimizer.')
    parser.add_argument('--max_grad_norm_clip',
                        type=float,
                        default=1.0,
                        help='Maximum gradient norm (for gradient clipping)')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=7.5e-6,
                        help='The initial learning rate for the AdamW optimizer.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='proportion of activations to drop in dropout layers')
    parser.add_argument('--prompt_loss_weight',
                        type=float,
                        default=0.0,
                        help='Relative weight of tokens with the "prompt" token type ID '
                        'during backpropagation.')


def add_run_args(parser: argparse.ArgumentParser):
    """Adds arguments used at runtime to an argument parser object

    Args:
        parser (argparse.ArgumentParser): The argument parser object to add arguments to
    """
    parser.add_argument('--data_dir', type=str, help='Path to a directory containing HDF5 files of pre-tokenized text')
    parser.add_argument('--steps', type=int, default=800, help='Number of training steps to take')
    parser.add_argument('--min_eval_acc',
                        type=float,
                        default=0.0,
                        help='Minimum threshold for evaluation accuracy of a trained model. only for testing.')
    parser.add_argument('--subsample_eval',
                        type=float,
                        default=0.1,
                        help='Proportion of the evaluation set to use for evaluation. '
                        'Setting a smaller poportion helps speed up evauation.')
    parser.add_argument('--checkpoint_name',
                        type=str,
                        default='checkpoint.pt',
                        help='Path where the final trained checkpoint will be saved.')


def get_model_trace_inputs(args: argparse.Namespace) -> Tuple[Any]:
    """Get input tensors to use for tracing the model.
    
    Since they're only used for tracing, these tensors are composed of dummy data.

    Args:
        args (argparse.Namespace): Parsed command line arguments

    Returns:
        Tuple[Any]: Inputs to use for tracing
    """

    # Make input_ids
    input_ids = torch.randint(0, 5000, (args.batch_size, args.max_seq_length), dtype=torch.int32)
    input_ids = samba.from_torch_tensor(input_ids, name="input_ids")

    # Make position_ids
    position_ids = torch.arange(args.max_seq_length)
    position_ids = position_ids.short()
    position_ids = samba.from_torch_tensor(position_ids.unsqueeze(0).expand(input_ids.shape), name='input_position_ids')

    # Make labels
    labels = torch.ones(args.batch_size, args.max_seq_length, dtype=torch.int16)
    labels = samba.from_torch_tensor(labels, name='labels')

    # Prepare the tracing items
    tracing_inputs = (input_ids, None, None, None, position_ids, None, None, None, None, labels)
    return tracing_inputs


def get_runtime_inputs(torch_input: Sequence[Optional[samba.SambaTensor]]) -> Sequence[Optional[samba.SambaTensor]]:
    """Given inputs from the dataset, create inputs for samba.session.run.

    These inputs must be the same dtype and shape as the compile inputs

    Args:
        inputs (Dict[str, List[Any]]): Inputs from the data loader

    Returns:
        Sequence[Optional[samba.SambaTensor]]: The named input tensors to use in running the model
    """
    torch_input = torch_input if len(torch_input) == 4 else ([torch_input[0]] + [None] + torch_input[1:])
    input_ids, attention_mask, position_ids, labels = torch_input

    # Create input IDs
    input_ids = samba.from_torch_tensor(input_ids.int(), name="input_ids")

    # Create position IDs
    position_ids = samba.from_torch_tensor(position_ids, name='input_position_ids')

    # Create labels
    labels = samba.from_torch_tensor(labels, name='labels')

    # Optionally add attention mask
    if attention_mask is not None:
        attention_mask = samba.from_torch_tensor(attention_mask, name="attention_mask")
        return [input_ids, attention_mask, position_ids, labels]
    else:
        return [input_ids, position_ids, labels]


def pad_tensor(t: torch.Tensor, batch_size: int, pad_val: float) -> torch.Tensor:
    """Pad tensor on the first dimension (batch dimension) to complete batch_size.

    Args:
        t (torch.Tensor): The tensor to pad
        batch_size (int): The size to pad to
        pad_val (float): The value to pad with

    Returns:
        torch.Tensor: The padded tensor
    """
    shape = list(t.shape)
    shape[0] = batch_size - shape[0]
    return torch.cat([t, torch.full(shape, pad_val, dtype=t.dtype)], dim=0)


def prepare_inputs(args: argparse.Namespace,
                   inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Sequence[Optional[torch.Tensor]], torch.Tensor]:
    """Prepare a batch of torch tensors from the data loader for passing into the model.

    This involves creating position IDs, shifting over the input_ids by 1 position
    to create labels, and creating target_token_type_ids to match the labels. The final
    tuple of tensors should match the shape of the inputs used for tracing.

    Args:
        args (argparse.Namespace): The parsed command line args
        inputs (Tuple[torch.Tensor, torch.Tensor]): The inputs from the data loader

    Returns:
        Tuple[Sequence[Optional[torch.Tensor]], torch.Tensor]: The inputs for run, and the token type IDs
    """
    input_ids = inputs[0].int()
    batch_size = input_ids.shape[0]

    # The train dataloader does not contain the following entries
    position_ids = torch.arange(args.max_seq_length)
    position_ids = position_ids.unsqueeze(0).expand(input_ids.shape)
    position_ids = position_ids.short()

    # Prepare the attention mask for the Hugging Face Module
    # set the labels to the input_ids, to be modified in the GPT model
    labels = input_ids.short()
    labels = labels[..., 1:]
    labels = torch.cat((labels, torch.ones([labels.shape[0], 1], dtype=labels.dtype) * -100), dim=1)

    # Construct the token type IDs
    token_type_ids = inputs[1].int()
    target_token_type_ids = token_type_ids[..., 1:]
    target_token_type_ids = torch.cat(
        (target_token_type_ids,
         torch.ones([target_token_type_ids.shape[0], 1], dtype=target_token_type_ids.dtype) * PADDING_TOKEN_TYPE_ID),
        dim=1)

    # Pad inputs to match the batch size
    if batch_size < args.batch_size:
        input_ids = pad_tensor(input_ids, args.batch_size, 0)
        position_ids = pad_tensor(position_ids, args.batch_size, 0)
        labels = pad_tensor(labels, args.batch_size, -100)
        target_token_type_ids = pad_tensor(target_token_type_ids, args.batch_size, PADDING_TOKEN_TYPE_ID)
    traced_inputs = (input_ids, None, None, None, position_ids, None, None, None, None, labels)

    return traced_inputs, target_token_type_ids


def compute_loss_scale(args: argparse.Namespace, targets: torch.Tensor, target_token_type_ids: torch.Tensor,
                       output_dtype: Union[torch.dtype, str]) -> torch.Tensor:
    """Compute the scale factor of the loss, depending on the labels indicated by the padding tokens/ignored indices.

    This is used to compute the correct value of the loss gradient to start backpropagation

    Args:
        args (argparse.Namespace): The parsed command line args
        targets (torch.Tensor): The target tensor for training
        target_token_type_ids (torch.Tensor): The token type IDs of the target tensor
        output_dtype (Union[torch.dtype, str]): The data type to output the loss scale in

    Returns:
        torch.Tensor: The scale factor of the loss
    """

    # ignore_index = -100 by default
    grad_scale_not_ignored = ~targets.eq(-100)
    # token_type_id = 2 identifies padding <eos> tokens
    grad_scale_not_ignored[target_token_type_ids.eq(PADDING_TOKEN_TYPE_ID)] = False
    grad_scale = grad_scale_not_ignored.float()
    # token_type_id = 0 identifies prompt tokens
    grad_scale[target_token_type_ids.eq(PROMPT_TOKEN_TYPE_ID)] *= args.prompt_loss_weight
    # normalize so that grad_scales sum to 1
    grad_scale /= torch.sum(grad_scale)
    loss_scale = grad_scale.bfloat16().to(output_dtype).flatten()
    return loss_scale


def model_step(args: argparse.Namespace, model: nn.Module, inputs: List[torch.Tensor],
               target_token_type_ids: torch.Tensor, traced_outputs: List[samba.SambaTensor]) -> torch.Tensor:
    """Take one training step on RDU

    Args:
        args (argparse.Namespace): The parsed command line arguments
        model (nn.Module): The model instance
        inputs (List[torch.Tensor]): The inputs for this step
        target_token_type_ids (torch.Tensor): The token type IDs
        traced_outputs (List[samba.SambaTensor]): The outputs of the model from tracing

    Returns:
        torch.Tensor: The loss for this step
    """
    inputs = [ipt for ipt in inputs if ipt is not None]
    learning_rate = args.learning_rate
    dropout_rate = args.dropout
    hyper_dict = {'lr': learning_rate}
    dropout_dict = {'p': dropout_rate}

    hyperparam_dict = {**hyper_dict, **dropout_dict}

    # Compute loss scale
    loss_scale = compute_loss_scale(args, inputs[-1], target_token_type_ids, model.output_tensors[0].dtype)

    # Convert input tensors to SambaTensor
    inputs_this_step = get_runtime_inputs(inputs)

    # Set the gradient of the output
    traced_outputs[0].sn_grad = loss_scale

    outputs = samba.session.run(inputs_this_step,
                                traced_outputs,
                                hyperparam_dict=hyperparam_dict,
                                section_types=['FWD', 'BCKWD', 'GRADNORM', 'OPT'])

    samba_loss = outputs[0]
    loss = samba.to_torch(samba_loss).float()
    loss *= loss_scale.float()
    loss = loss.sum()
    return loss


def exact_match_accuracy(labels_list: List[torch.Tensor], preds_list: List[torch.Tensor]) -> float:
    """Compute the exact match accuracy between the true labels and predicted labels

    Args:
        labels_list (List[torch.Tensor]): The true labels
        preds_list (List[torch.Tensor]): The predicted labels

    Returns:
        float: The exact match accuracy
    """

    assert len(labels_list) == len(preds_list)
    total = 0
    match = 0
    for (label, pred) in zip(labels_list, preds_list):
        total += 1
        if torch.equal(label, pred):
            match += 1
    return 1.0 * match / total


def evaluate(args: argparse.Namespace, model: nn.Module, traced_outputs: List[samba.SambaTensor]) -> Tuple[float]:
    """Evaluate the model's performance on RDU

    Args:
        args (argparse.Namespace): The parsed command line args
        model (nn.Module): The model instance
        traced_outputs (List[samba.SambaTensor]): The outputs of the model from tracing

    Returns:
        Tuple[float]: The average evaluation loss & the evaluation accuracy
    """

    total_eval_loss = []
    labels_list = []
    preds_list = []

    eval_dataloaders = get_eval_iterators(args)

    with build_progress_bar(eval_dataloaders) as pbar:
        for eval_iter in eval_dataloaders:
            for step, batch in enumerate(eval_iter):
                inputs, target_token_type_ids = prepare_inputs(args, batch)

                inputs = [t for t in inputs if t is not None]
                hyperparam_dict = {}
                hyperparam_dict['lr'] = 0.0
                hyperparam_dict['p'] = 0.0

                # prepare current step input & perform model fwd
                inputs_this_step = get_runtime_inputs(inputs)
                outputs = samba.session.run(inputs_this_step,
                                            traced_outputs,
                                            hyperparam_dict=hyperparam_dict,
                                            section_types=["FWD"])
                logits = samba.to_torch(outputs[1])

                # Compute predictions for each sample
                for sample in range(args.batch_size):
                    targets = inputs_this_step[-1][sample]
                    logit = logits[sample]
                    preds = torch.argmax(logit, axis=-1)

                    # If completion tokens are missing, this is probably an incomplete batch from
                    # the end of the dataloader. We exclude it from the evaluation
                    if COMPLETION_TOKEN_TYPE_ID not in target_token_type_ids[sample]:
                        continue
                    idx = (target_token_type_ids[sample] == COMPLETION_TOKEN_TYPE_ID)

                    # Make sure label and pred have the same dtype
                    labels_list.append(samba.to_torch(targets[idx]).short())
                    preds_list.append(samba.to_torch(preds[idx]).short())

                # Compute loss
                loss_scale = compute_loss_scale(args, inputs[-1], target_token_type_ids, model.output_tensors[0].dtype)
                samba_loss = outputs[0]
                loss = samba.to_torch(samba_loss).float()
                loss *= loss_scale.float()
                loss = loss.sum()
                loss = samba.to_torch(loss)
                total_eval_loss.append(loss.item())
                pbar.update(1)

    # Compute exact match accuracy
    eval_acc = exact_match_accuracy(labels_list, preds_list)
    return sum(total_eval_loss) / len(total_eval_loss), eval_acc


def get_optimizers(args: argparse.Namespace, model: torch.nn.Module) -> List[torch.optim.Optimizer]:
    """Construct the optimizers

    Create separate optimizers for Embeddings, parameters that need weight decay, and parameters that do not

    Args:
        args (argparse.Namespace): The parsed command line arguments
        model (torch.nn.Module): The model instance

    Returns:
        List[torch.optim.Optimizer]: The optimizers
    """
    emb_modules = [module for module in model.modules() if isinstance(module, torch.nn.Embedding)]
    emb_params = OrderedSet(itertools.chain(*[emb.parameters() for emb in emb_modules]))
    other_params = OrderedSet([(name, param) for name, param in model.named_parameters() if param not in emb_params])

    # Exclude weight decay from bias & layernorm parameters
    no_decay = ["bias"]
    for name, params in model.named_parameters():
        if "ln" in name or "layernorm" in name or "layer_norm" in name:
            no_decay.append(name)
    params_w_weight_decay = OrderedSet([(n, p) for n, p in other_params if not any(nd in n for nd in no_decay)])
    params_wo_weight_decay = OrderedSet([(n, p) for n, p in other_params if any(nd in n for nd in no_decay)])

    emb_optim = samba.optim.AdamW(emb_params,
                                  lr=args.learning_rate,
                                  betas=(0.9, 0.997),
                                  eps=1e-8,
                                  weight_decay=args.weight_decay,
                                  max_grad_norm=args.max_grad_norm_clip)
    opt_w_weight_decay = samba.optim.AdamW([param for (name, param) in params_w_weight_decay],
                                           lr=args.learning_rate,
                                           betas=(0.9, 0.997),
                                           weight_decay=args.weight_decay,
                                           max_grad_norm=args.max_grad_norm_clip)
    opt_wo_weight_decay = samba.optim.AdamW([param for (name, param) in params_wo_weight_decay],
                                            lr=args.learning_rate,
                                            betas=(0.9, 0.997),
                                            weight_decay=0,
                                            max_grad_norm=args.max_grad_norm_clip)

    return [emb_optim, opt_w_weight_decay, opt_wo_weight_decay]


def patch_model(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    """Patch the Hugging Face model to make it more efficient when running on RDU.

    Args:
        model (nn.Module): The Hugging Face model instance
        args (argparse.Namespace): The parsed command line args

    Returns:
        nn.Module: The patched model instance
    """
    return gpt2_patch_helper(model)


def get_epoch_train_iterators(args: argparse.Namespace) -> List[torch.utils.data.DataLoader]:
    """Get a list of dataloaders that will iterate over all of the files in the training dataset.

    Args:
        args (argparse.Namespace): The parsed command line arguments

    Returns:
        List[torch.utils.data.Dataloader]: The dataloaders for this set of files
    """
    files = [
        os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
        if os.path.isfile(os.path.join(args.data_dir, f)) and ('train' in f)
    ]
    files.sort()
    len(files)
    dataloaders = []
    for data_file in files:
        train_data = PretrainingGenerativeDataset(input_file=data_file)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)

        dataloaders.append(train_dataloader)
    return dataloaders


def build_progress_bar(dataloaders: List[DataLoader]) -> tqdm:
    """Construct a combined tqdm instance (progress bar) for a list of data loaders

    Args:
        dataloaders (List[DataLoader]): The dataloaders to iterate over

    Returns:
        tqdm: The tqdm instance
    """
    return tqdm(total=sum([len(dataloader) for dataloader in dataloaders]))


def get_eval_iterators(args: argparse.Namespace) -> List[DataLoader]:
    """Get a list of dataloaders that will iterate over all of the files in the evaluation dataset.

    Will also subsample evaluation files according to the provided subsample proportion

    Args:
        args (argparse.Namespace): The parsed command line args

    Returns:
        List[DataLoader]: The list of dataloaders
    """
    files = [
        os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
        if os.path.isfile(os.path.join(args.data_dir, f)) and ('dev' in f or 'test' in f)
    ]
    files.sort()
    num_files = len(files)

    assert 0.0 <= args.subsample_eval <= 1.0, "Subsample eval should be between [0, 1.0]"
    # Subsample the validation file
    num_files_to_evaluate = int(math.ceil(args.subsample_eval * num_files))
    assert num_files_to_evaluate > 0, "Must have at least 1 eval file! " + \
        "Try increasing args.subsample_eval to a large value (max 1.0) or " + \
        "check the file dir to see if the files are missing"

    files_to_eval = random.sample(range(num_files), k=num_files_to_evaluate)

    dataloaders = []
    for k, data_file in enumerate(files):
        if k not in files_to_eval:
            continue
        eval_data = PretrainingGenerativeDataset(input_file=data_file)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)
        dataloaders.append(eval_dataloader)

    num_samples = sum([len(dataloader) for dataloader in dataloaders])
    print(
        f"Evaluating on {num_files_to_evaluate} files out of {num_files} with {num_samples} total samples in the evaluation dataset",
        flush=True)

    return dataloaders


def train(args: argparse.Namespace, model: nn.Module, traced_outputs: List[samba.SambaTensor]):
    """Perform the training procedure

    Args:
        args (argparse.Namespace): The parsed command line arguments
        model (nn.Module): The model instance
        traced_outputs (List[samba.SambaTensor]): The outputs of the model from tracing
    """
    eval_total_loss = None
    epoch = 1
    total_steps_taken = 0
    training_done = False

    # Depend on the provided step count rather than epochs
    while not training_done:
        print(f"Training Epoch {epoch}")
        dataloaders = get_epoch_train_iterators(args)
        with build_progress_bar(dataloaders) as pbar:

            for dataloader in dataloaders:

                # Break out if steps exceed specified steps
                if total_steps_taken >= args.steps:
                    if not training_done:
                        print(f"Finished training at {total_steps_taken} steps!")
                    training_done = True
                    break

                for batch in dataloader:

                    # Break out if steps exceed specified steps
                    if total_steps_taken >= args.steps:
                        if not training_done:
                            print(f"Finished training at {total_steps_taken} steps!")
                        training_done = True
                        break

                    inputs, target_token_type_ids = prepare_inputs(args, batch)
                    inputs = [t for t in inputs if t is not None]

                    # Take one training step
                    loss = model_step(args, model, inputs, target_token_type_ids, traced_outputs)
                    train_loss = loss.item()
                    total_steps_taken += 1
                    pbar.update(1)
                    pbar.set_description(f"Training loss: {train_loss}")

        if not training_done:
            eval_total_loss, eval_acc = evaluate(args, model, traced_outputs)
            print(
                f"Evaluation Results At Step {total_steps_taken} : Total Loss: {eval_total_loss}, Eval acc: {eval_acc}")
            epoch += 1
    print("Finished training")
    # Evaluate
    eval_total_loss, eval_acc = evaluate(args, model, traced_outputs)

    print(f"Final eval total loss: {eval_total_loss}\nFinal eval accuracy: {eval_acc}")

    if args.min_eval_acc > 0.0:
        assert eval_acc >= args.min_eval_acc, f"Obtained eval_acc={eval_acc}, Expected Minimum eval_acc={args.min_eval_acc}"

    # Save checkpoint
    save_checkpoint(model, args.checkpoint_name)


def main(argv: List[str]) -> None:
    # Parse the args
    args = parse_app_args(argv=argv, common_parser_fn=add_common_args, run_parser_fn=add_run_args)

    # Download the model from Hugging Face
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        model.training = True
    elif args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        # Read dropout rate from config
        args.dropout = config.resid_pdrop
        model = AutoModelForCausalLM.from_config(config)
    else:
        raise RuntimeError("Must provide --model_name_or_path or --config_name")

    if not args.inference:
        model = model.train()
    else:
        model = model.eval()

    # Patch the model here
    model = patch_model(model, args)

    samba.from_torch_model_(model)

    # Make the tracing inputs
    inputs = get_model_trace_inputs(args)

    # Make the optimizer
    optims = get_optimizers(args, model)

    if args.command == 'compile':
        samba.session.compile(model, inputs, optims, name='hf_transformer', init_output_grads=True)
    elif args.command == 'run':
        traced_outputs = utils.trace_graph(model, inputs, optims, pef=args.pef, init_output_grads=True)
        train(args, model, traced_outputs)


if __name__ == "__main__":
    main(sys.argv[1:])
