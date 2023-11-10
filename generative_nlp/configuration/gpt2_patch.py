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
from typing import Tuple

import torch
import torch.nn as nn
import transformers
from torch.nn import CrossEntropyLoss

import sambaflow.samba as samba
from sambaflow.logging import samba_logger
from sambaflow.samba import SambaTensor
from sambaflow.samba.directives import op_fusion
from sambaflow.samba.nn.parameter import SambaParameter


def gpt2_patch_helper(model: nn.Module) -> nn.Module:
    """Patches module forward calls within a gpt2-based transformer model to
    make them more efficient in the SambaNova compiler

    Returns:
        nn.Module: The patched transformer model instance
    """
    model.return_logits = True
    model.no_index_select_patch = True
    model.return_cache = False

    model.forward = gpt2_head_forward.__get__(model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel)

    for ind, layers in enumerate(model.transformer.h):
        layers.mlp.act = nn.GELU()
        size = (1, 1, layers.attn.bias.shape[-1], layers.attn.bias.shape[-2])
        layers.attn.bias = layers.attn.bias.expand(size)
        layers.attn.c_attn.weight = transpose_weight(layers.attn.c_attn.weight)
        layers.attn.c_proj.weight = transpose_weight(layers.attn.c_proj.weight)
        layers.mlp.c_fc.weight = transpose_weight(layers.mlp.c_fc.weight)
        layers.mlp.c_proj.weight = transpose_weight(layers.mlp.c_proj.weight)

        layers.attn.c_attn.forward = gpt2_conv1d_forward.__get__(layers.attn.c_attn, transformers.modeling_utils.Conv1D)
        layers.attn.c_proj.forward = gpt2_conv1d_forward.__get__(layers.attn.c_proj, transformers.modeling_utils.Conv1D)
        layers.mlp.c_fc.forward = gpt2_conv1d_forward.__get__(layers.mlp.c_fc, transformers.modeling_utils.Conv1D)
        layers.mlp.c_proj.forward = gpt2_conv1d_forward.__get__(layers.mlp.c_proj, transformers.modeling_utils.Conv1D)

        op_fusion(layers, 'Encoder')

    return model


def gpt2_head_forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
) -> Tuple[torch.Tensor]:
    """Patches out the index_select usage in the Huggingface GPT2 head forward to improve performance on RDU.
    Default values for arguments here are usually fine

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, optional):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, optional):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, optional):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, optional):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, optional):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        encoder_hidden_states (`torch.FloatTensor`, optional): Hidden states to use in cross attention. Defaults to None.
        encoder_attention_mask (`torch.FloatTensor`, optional): Attention mask to use in cross attention. Defaults to None.
        labels (`torch.LongTensor`, optional): labels to use in CrossEntropyLoss. Defaults to None.
        use_cache (`bool`, optional):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, optional):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, optional):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, optional):
            Whether or not to return a Huggingface ModelOutput object instead of a plain Tuple.

    Returns:
        Tuple[torch.Tensor]: Output tensors of the GPT2 head forward
    """
    use_cache = self.config.use_cache
    samba_logger.warning("Warning: Patched GPT2LMhead function used", log_limit=1)
    if "past" in kwargs:
        warnings.warn(
            "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
            FutureWarning,
        )
        past_key_values = kwargs.pop("past")
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]
    lm_logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if (self.no_index_select_patch):
            samba_logger.warning("Warning: Skipping index select", log_limit=1)
            shift_logits = lm_logits
            shift_labels = labels
        else:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        outputs = (loss, )
        if self.return_logits:
            outputs += (lm_logits * 1, )
    else:
        samba_logger.warning("Warning: No labels provided, returning logits")
        outputs = (lm_logits, )

    if hasattr(self, 'return_intermediate_vals') and self.return_intermediate_vals:
        samba_logger.warning("Warning: Returning intermediate values", log_limit=1)
        if not self.return_logits:
            # return logits if they haven't already been specified
            outputs += (lm_logits * 1, )
        outputs += (hidden_states * 1, )
    return outputs


def gpt2_conv1d_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Patches Conv1D function to use optimized addmm instead

    Args:
        x (torch.Tensor): The input tensor, required

    Returns:
        torch.Tensor: The output tensor
    """
    samba_logger.warning("Warning: Patched gpt2_conv1d_forward function called instead of conv1d.forward", log_limit=1)
    x = samba.addmm(self.bias, x, self.weight, is_transposed=True)
    return x


def transpose_weight(tensor: torch.Tensor):
    """Transposes SambaParameters in the model inplace so they have the right shape for the addmm patch

    Args:
        tensor (torch.Tensor): The parameter to transpose

    Returns:
        torch.Tensor: The transposed parameter
    """
    if isinstance(tensor, SambaParameter):

        def materialize_transpose(this_tensor: SambaTensor, original_tensor=tensor) -> torch.Tensor:
            return original_tensor.torch_tensor().transpose(1, 0)

        return SambaParameter(shape=reversed(tensor.shape),
                              dtype=tensor.dtype,
                              materializer=materialize_transpose,
                              requires_grad=tensor.requires_grad)
    elif isinstance(tensor, torch.Tensor):
        return nn.Parameter(data=tensor.transpose(1, 0), requires_grad=tensor.requires_grad)
    else:
        assert False, 'Unhandled tensor type'
