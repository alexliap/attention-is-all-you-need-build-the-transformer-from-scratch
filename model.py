"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    out = {val:i for val, i in zip(specials, range(len(specials)))}

    i = len(out)
    for item in sentences:
        split_item = item.split(" ")
        for word in split_item:
            if word in out.keys():
                continue
            else:
                out[word] = i
        
            i += 1

    return out

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    return {id:val for val, id in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    out = []
    if not sentence.strip():
        return out
        
    for item in sentence.split(" "):
        if item not in token_to_id.keys():
            out.append(token_to_id[unk_token])
        else:
            out.append(token_to_id[item])

    return out

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    out = []
    for idx in ids:
        out.append(id_to_token[idx])

    return out

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    if len(ids) <= max_len:
        return ids + [pad_id] * (max_len - len(ids))
    elif len(ids) > max_len:
        return ids[:max_len]

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    out = None
    for item in padded_sequences:
        if out is None:
            out = torch.tensor(item)
        else:
            out = torch.vstack((out, torch.tensor(item)))
    return out.reshape(len(padded_sequences), -1)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    return embeddings*math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    i = torch.arange(d_model // 2)
    return torch.exp(2*i * (- math.log(10000)/d_model))

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    pe[:, 0::2] = torch.sin(position*div_term)

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    pe[:, 1::2] = torch.cos(position*div_term)

    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    out = torch.zeros((max_len, d_model))

    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)

    out = fill_even_indices_with_sin(out, position, div_term)
    out = fill_odd_indices_with_cos(out, position, div_term)

    return out

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    _, L, _ = embedded_batch.shape
    positional_encoding = positional_encoding[:L].reshape(1, L,
                                                          positional_encoding.shape[1])

    return embedded_batch + positional_encoding

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    B, L = token_ids.shape
    mask = torch.ones(size=token_ids.size(), dtype=torch.bool)
    pad_idxs = torch.where(token_ids == pad_id)
    mask[pad_idxs] = False
    return mask.view(B, 1, 1, L)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    mask = torch.ones(size=(seq_len, seq_len), dtype=torch.bool)
    for i in range(seq_len):
        mask[:i, i] = False

    return mask.view(1, 1, seq_len, seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    return torch.logical_and(padding_mask, causal_mask)

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    return query @ key.transpose(-1, -2)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return scores.masked_fill(~mask, - torch.inf)

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(masked_scores):
    fully_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)
    safe = masked_scores.masked_fill(fully_masked, 0.0)
    w = F.softmax(safe, dim=-1)
    return w.masked_fill(fully_masked, 0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    scores = compute_raw_attention_scores(query, key)
    scores = scale_attention_scores(scores, key.shape[-1])
    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)

    scores = softmax_attention_weights(scores)
    return apply_attention_weights_to_values(scores, value), scores

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.shape
    return tensor.view(B, L, num_heads, -1)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(dim0=1, dim1=2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B, num_heads, L, d_k = multi_head_tensor.shape
    multi_head_tensor = multi_head_tensor.transpose(dim0=1, dim1=2)
    return multi_head_tensor.reshape(B, L, -1)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    out = x @ weight.T
    if bias is not None:
        return out + bias
    return out

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)
    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q_h = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_h = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_h = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))
    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    out = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(out, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)

    q, k, v = split_qkv_into_heads(q, k, v, num_heads)
    context, _ = multi_head_scaled_dot_product_attention(q, k, v, mask)

    return merge_heads_and_project_output(context, w_o, None)

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    ffn_out = x @ w1 + b1
    return F.relu(ffn_out)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    ffn_out = hidden @ w2 + b2
    return ffn_out

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    out = apply_ffn_second_linear(hidden, w2, b2)
    return out

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, var = compute_layer_norm_mean_and_variance(x)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_hat + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    ln = normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta)
    return ln

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    return (x * keep_mask) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    out = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    x = apply_residual_add_and_norm(x, out, gamma, beta, eps=1e-5)
    return x

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    out = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    x = apply_residual_add_and_norm(x, out, gamma, beta)
    return x

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    p = layer_params

    x = encoder_layer_self_attention_sublayer(
        x,
        p["w_q"], p["w_k"], p["w_v"], p["w_o"],
        p["attn_gamma"], p["attn_beta"],
        num_heads, src_mask,
    )

    x = encoder_layer_feed_forward_sublayer(
        x,
        p["w1"], p["b1"], p["w2"], p["b2"],
        p["ffn_gamma"], p["ffn_beta"],
    )

    return x

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    for layer in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer, num_heads, src_mask)

    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    out = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    y = apply_residual_add_and_norm(y, out, gamma, beta)
    return y

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask[:, None, None, :]
    out = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)
    y = apply_residual_add_and_norm(y, out, gamma, beta)
    return y

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    out = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, out, gamma, beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN.

    layer_params keys (all torch tensors):
      masked self-attention : w_q_self, w_k_self, w_v_self, w_o_self, self_gamma, self_beta
      cross-attention       : w_q_cross, w_k_cross, w_v_cross, w_o_cross, cross_gamma, cross_beta
      feed-forward          : w1, b1, w2, b2, ffn_gamma, ffn_beta
    """
    p = layer_params

    y = decoder_layer_masked_self_attention_sublayer(
        y,
        p["w_q_self"], p["w_k_self"], p["w_v_self"], p["w_o_self"],
        p["self_gamma"], p["self_beta"],
        num_heads, tgt_mask,
    )

    y = decoder_layer_cross_attention_sublayer(
        y, encoder_output,
        p["w_q_cross"], p["w_k_cross"], p["w_v_cross"], p["w_o_cross"],
        p["cross_gamma"], p["cross_beta"],
        num_heads, src_mask,
    )

    y = decoder_layer_feed_forward_sublayer(
        y,
        p["w1"], p["b1"], p["w2"], p["b2"],
        p["ffn_gamma"], p["ffn_beta"],
    )

    return y

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    for layer in decoder_layer_params_list:
        out = assemble_decoder_layer(y, encoder_output, layer, num_heads, src_mask, tgt_mask)
        y = out

    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    return F.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    emb = model_params["token_embedding"]            # (vocab_size, d_model)
    d_model = emb.shape[1]

    max_len = max(src_ids.shape[1], tgt_ids.shape[1])
    pe = build_sinusoidal_positional_encoding(max_len, d_model)

    # embed, scale, add positional encoding
    x = scale_embeddings_by_sqrt_d_model(emb[src_ids], d_model)
    x = add_positional_encoding_to_embeddings(x, pe)
    y = scale_embeddings_by_sqrt_d_model(emb[tgt_ids], d_model)
    y = add_positional_encoding_to_embeddings(y, pe)

    # masks
    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_mask = combine_padding_and_causal_masks(
        build_padding_mask(tgt_ids, pad_id),
        build_causal_mask(tgt_ids.shape[1]),
    )

    # encoder, then decoder
    enc = stack_encoder_layers(x, model_params["encoder_layers"], num_heads, src_mask)
    dec = stack_decoder_layers(y, enc, model_params["decoder_layers"], num_heads, src_mask, tgt_mask)

    # project to vocab, then log probs
    logits = apply_final_output_projection(dec, model_params["output_projection"])
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    def weight(fan_in, fan_out):
        std = math.sqrt(2.0 / (fan_in + fan_out))  # Xavier-style scale
        w = torch.randn(fan_in, fan_out, dtype=torch.float32) * std
        return w.requires_grad_(True)

    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.float32, requires_grad=True)

    def ones(*shape):
        return torch.ones(*shape, dtype=torch.float32, requires_grad=True)

    return {
        "w_q": weight(d_model, d_model),
        "w_k": weight(d_model, d_model),
        "w_v": weight(d_model, d_model),
        "w_o": weight(d_model, d_model),
        "w1": weight(d_model, d_ff),
        "b1": zeros(d_ff),
        "w2": weight(d_ff, d_model),
        "b2": zeros(d_model),
        "attn_gamma": ones(d_model),
        "attn_beta": zeros(d_model),
        "ffn_gamma": ones(d_model),
        "ffn_beta": zeros(d_model),
    }

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    def weight(fan_in, fan_out):
        std = math.sqrt(2.0 / (fan_in + fan_out))  # Xavier-style scale
        w = torch.randn(fan_in, fan_out, dtype=torch.float32) * std
        return w.requires_grad_(True)

    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.float32, requires_grad=True)

    def ones(*shape):
        return torch.ones(*shape, dtype=torch.float32, requires_grad=True)

    return {
        "w_q_self": weight(d_model, d_model),
        "w_k_self": weight(d_model, d_model),
        "w_v_self": weight(d_model, d_model),
        "w_o_self": weight(d_model, d_model),
        "w_q_cross": weight(d_model, d_model),
        "w_k_cross": weight(d_model, d_model),
        "w_v_cross": weight(d_model, d_model),
        "w_o_cross": weight(d_model, d_model),
        "w1": weight(d_model, d_ff),
        "b1": zeros(d_ff),
        "w2": weight(d_ff, d_model),
        "b2": zeros(d_model),
        "self_gamma": ones(d_model),
        "self_beta": zeros(d_model),
        "cross_gamma": ones(d_model),
        "cross_beta": zeros(d_model),
        "ffn_gamma": ones(d_model),
        "ffn_beta": zeros(d_model),
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    def weight(fan_in, fan_out):
        std = math.sqrt(2.0 / (fan_in + fan_out))  # Xavier-style scale
        w = torch.randn(fan_in, fan_out, dtype=torch.float32) * std
        return w.requires_grad_(True)

    tgt_embedding = weight(vocab_size, d_model)
    return {
        "src_embedding": weight(vocab_size, d_model),
        "tgt_embedding": tgt_embedding,
        "output_projection": tgt_embedding if tie_weights else weight(vocab_size, d_model)
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    params = []
    seen = set()

    def add(tensor):
        if tensor.requires_grad and id(tensor) not in seen:
            seen.add(id(tensor))
            params.append(tensor)

    for layer in encoder_layer_params:
        for tensor in layer.values():
            add(tensor)
    for layer in decoder_layer_params:
        for tensor in layer.values():
            add(tensor)
    for tensor in embedding_params.values():
        add(tensor)

    return params

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    device = target_ids.device
    start_tensor = torch.tensor([start_token_id] * target_ids.shape[0]).view(-1, 1)

    return torch.hstack((start_tensor, target_ids[:, :-1]))

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    scalar = (1/math.sqrt(d_model))
    if step == 0:
        return scalar
    option = min((1/math.sqrt(step)), step*(warmup_steps**(-3/2)))
    return scalar*option

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    return torch.ones(size=shape) * (epsilon/(vocab_size-2))

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    B, T, _ = smoothed_distribution.shape
    out = smoothed_distribution.clone()
    out.scatter_(2, gold_token_ids.reshape(B, T, 1), confidence)
    return out

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    B, T, V = smoothed_distribution.shape
    smoothed_distribution[:, :, pad_id] = 0
    smoothed_distribution[gold_token_ids == pad_id, :] = 0

    return smoothed_distribution

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    return torch.sum(-log_probabilities * smoothed_distribution)

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    if torch.sum(gold_token_ids != pad_id) == 0:
        return total_loss
    return total_loss / torch.sum(gold_token_ids != pad_id)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    acc = torch.argmax(log_probabilities, dim=-1) == gold_token_ids
    pad_tokens = gold_token_ids == pad_id

    if torch.sum(pad_tokens) == pad_tokens.shape[0] * pad_tokens.shape[1]:
        return torch.tensor(0)

    nom = torch.sum(acc[~pad_tokens])
    denom = log_probabilities.shape[0] * log_probabilities.shape[1] - torch.sum(pad_tokens)

    return nom / denom

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    m = [torch.zeros_like(item, requires_grad=False) for item in parameter_list]
    v = [torch.zeros_like(item, requires_grad=False) for item in parameter_list]
    return {
        "m": m,
        "v": v,
        "t": 0
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    return beta1 * m_prev + (1 - beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    return beta2 * v_prev + (1 - beta2) * grad ** 2

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    m_hat = m_t / (1-beta1**step)
    v_hat = v_t / (1-beta2**step)
    return m_hat, v_hat

# Step 68 - compute_adam_parameter_update
import torch

def compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon):
    """Return delta = learning_rate * m_hat / (sqrt(v_hat) + epsilon); the caller subtracts it."""
    return learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    optimizer_state["t"] += 1
    m_prev = optimizer_state["m"]
    v_prev = optimizer_state["v"]
    step = optimizer_state["t"]

    for i in range(len(parameter_list)):
        if parameter_list[i].requires_grad and parameter_list[i].grad is not None:
            grad = parameter_list[i].grad

            m_t = update_adam_first_moment(m_prev[i], grad, beta1)
            v_t = update_adam_second_moment(v_prev[i], grad, beta2)

            m_hat, v_hat = apply_adam_bias_correction(m_t, v_t, beta1, beta2, step)
            delta = compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon)

            optimizer_state["m"][i] = m_t
            optimizer_state["v"][i] = v_t

            with torch.no_grad():
                parameter_list[i] -= delta

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    for i in range(len(parameter_list)):
        parameter_list[i].grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    pad_id = config["pad_id"]

    # teacher forcing: decoder input is the target shifted right with the start token
    decoder_input = shift_targets_right_with_start_token(tgt_batch, config["start_id"])

    # (B, T, V) log probabilities
    log_probs = run_transformer_forward(
        src_batch, decoder_input, model_params, config["num_heads"], pad_id
    )

    # smoothed target distribution built against the UNSHIFTED gold tokens
    smoothing = config["smoothing"]
    dist = build_uniform_smoothing_distribution(
        tuple(log_probs.shape), config["vocab_size"], smoothing
    ).to(device=log_probs.device, dtype=log_probs.dtype)
    dist = set_confidence_on_gold_tokens(dist, tgt_batch, 1.0 - smoothing)
    dist = zero_pad_column_and_pad_token_rows(dist, tgt_batch, pad_id)

    # summed loss, then mean over non-pad tokens
    total_loss = compute_label_smoothed_kl_loss(log_probs, dist)
    return average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    # zero grads
    zero_all_parameter_gradients(parameter_list)
    # compute loss
    loss = compute_batch_training_loss(src_batch, tgt_batch, model_params, config)
    # backward
    loss.backward()
    # look up Noam LR
    learning_rate = compute_noam_learning_rate(optimizer_state["t"], config["d_model"], config["warmup_steps"])
    # apply Adam step
    optimizer_state = apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate)

    return loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    loss_values = []
    for i in range(1, num_steps+1):
        batch_iter = i % len(batches) - 1
        loss = run_training_step_with_backprop(batches[batch_iter][0], batches[batch_iter][1], 
                                               parameter_list, model_params, optimizer_state, i, config)
        loss_values.append(loss)

    return loss_values

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    return torch.argmax(final_step_logits, dim=-1)

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

