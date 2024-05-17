# This file is created to patch librairies where we need some changes

import math
import sys
import pkgutil
try:
    from pkgutil import resolve_name
except:
    from pkgutil_resolve_name import resolve_name # Backport for older python versions
from typing import Optional, Tuple, List, Union
import collections.abc as container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.opt.modeling_opt import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import torch.nn.modules.linear as toch_nn_lin


# Fix unused problematic import from TimeSformer
toch_nn_lin._LinearWithBias = None
# Fix problematic import from TimeSformer (import 'container_abcs')
sys.modules['torch._six'] = sys.modules[__name__]

ALL_PATCHS = {}

def patch(target_full, strict=True):
    target, attribute = target_full.rsplit('.', 1)
    target = resolve_name(target)
    def _patcher(patched_function):
        if strict:
            assert getattr(target, attribute) is not None
        setattr(target, attribute, patched_function)
        ALL_PATCHS['target_full'] = patched_function
        return patched_function
    return _patcher

##### Transformer patchs #####

@patch('transformers.models.opt.modeling_opt.OPTAttention.forward')
def patched_opt_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    gating: bool = False, # [PATCH: allow gating]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Code from https://github.com/huggingface/transformers/blob/c5e29d4381d4b9739e6cb427adbca87fbb43a3ad/src/transformers/models/opt/modeling_opt.py#L155
    # Patch 1: remove size checks (required for compilation)
    # Patch 2: add gating (only works if model is wrapped by GatedAttentionLayer)
    # Patch 3: add faster Attention
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    ##### [BEGIN PATCH 3 : adding faster attention] #####
    if not gating and self.training:
        attn_output = F.scaled_dot_product_attention(
            query=self._shape(query_states, tgt_len, bsz) * self.head_dim**0.5, # Un-scale attention
            key=key_states,
            value=value_states,
            # dropout_p=self.dropout if not self.training else 0,
            dropout_p=0,
            # attn_mask=attention_mask,
            is_causal=True,
        ).transpose(1, 2).contiguous().reshape(bsz, tgt_len, self.embed_dim)
        attn_weights_reshaped = None # We don't compute explicitly the weights

    else: # Need to use old version with gating
        ##### [PATCH 3 : removed parts] #####

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        ##### [EDITED : remove check (reason: make dynamic compilation fail)] #####

        if attention_mask is not None:
            ##### [EDITED : remove check (reason: make dynamic compilation fail)] #####
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            ##### [EDITED : remove check (reason: make dynamic compilation fail)] #####
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        ##### [BEGIN PATCH : adding gating] #####
        if gating:
            assert len(attn_weights.shape) == 3, f"The shape of attn_weights is {attn_weights.shape}" # batch, .., ..
            attn_weights = torch.cat([
                attn_weights[:, :, :self.from_id],
                attn_weights[:, :, self.from_id:self.to_id] * self.gating_value,
                attn_weights[:, :, self.to_id:],
            ], dim=-1)
            # Normalize sum to 1
            eps = torch.tensor(1e-1, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = attn_weights / torch.maximum(attn_weights.sum(axis=-1, keepdim=True), eps)
        #####  [END EDITED SECTION]  #####

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        #### [EDITED : remove check (reason: make dynamic compilation fail)] #####

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    ##### [END PATCH 3] #####

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, past_key_value

@patch('transformers.models.opt.modeling_opt.OPTForCausalLM.forward')
def patched_opt_causalml_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    loss_smothing: float = 0, # [PATCH 1: add loss_smothing arg]
) -> Union[Tuple, CausalLMOutputWithPast]:
    # Code from https://github.com/huggingface/transformers/blob/35eac0df75c692c5b93c12f7eaf3279cab8bd7ce/src/transformers/models/opt/modeling_opt.py#L850
    # Patch 1: allow smoothing of cross entropy loss
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = self.lm_head(outputs[0]).contiguous()

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(label_smoothing=loss_smothing) # [PATCH 1: add label_smoothing arg]
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

@patch('transformers.models.llama.modeling_llama.LlamaForCausalLM.forward')
def patched_llama_causalml_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    loss_smothing: float = 0, # [PATCH 1: add loss_smothing arg]
) -> Union[Tuple, CausalLMOutputWithPast]:
    # Code from https://github.com/huggingface/transformers/blob/2489e380e45177eb3da108366f65b8108b2815c5/src/transformers/models/llama/modeling_llama.py#L645
    # Patch 1: allow smoothing of cross entropy loss

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(label_smoothing=loss_smothing) # [PATCH 1: add label_smoothing arg]
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

@patch('transformers.models.llama.modeling_llama.LlamaAttention.forward')
def patched_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    gating: bool = False, # [PATCH2: allow gating]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Code from https://github.com/huggingface/transformers/blob/4e8929dcbb9040f54f52d898a600260f338ef54f/src/transformers/models/llama/modeling_llama.py#L184
    # Patch 1: remove size checks (required for compilation)
    # Patch 2: add gating (only works if model is wrapped by GatedAttentionLayer)
    # Patch 3: add faster attention

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None


    ##### [BEGIN PATCH 3 : adding faster attention] #####
    if not gating and (self.training or not use_cache):
        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            # attn_mask=attention_mask,
            is_causal=True,
        ).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

    else: # Need to use old version with gating
        #### [BEGIN PATCH 3 : removed parts] #####
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        ##### [BEGIN PATCH 2 : adding gating] #####
        if gating:
            assert len(attn_weights.shape) == 4, f"The shape of attn_weights is {attn_weights.shape}" # batch, .., ..
            attn_weights = torch.cat([
                attn_weights[:, :, :, :self.from_id],
                attn_weights[:, :, :, self.from_id:self.to_id] * self.gating_value,
                attn_weights[:, :, :, self.to_id:],
            ], dim=-1)
            # Normalize sum to 1
            eps = torch.tensor(1e-3, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = attn_weights / torch.maximum(attn_weights.sum(axis=-1, keepdim=True), eps)
        #####  [END PATCH 2]  #####

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    ##### [END PATCH 3 : end patch 3 parts] #####

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
