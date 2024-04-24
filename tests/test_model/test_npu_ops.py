"""
TODO: add NPU CI
"""

import math
import random

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.model.modules.multi_head_attention import (
    AscendFlashAttention,
    CrossAttention,
    SelfAttention,
)
from internlm.model.ops.fusion_ops_import_helper import try_import_RMSNorm
from internlm.model.utils import pack_output_after_attn, unpack_qkv_before_attn
from internlm.utils.common import set_random_seed

RMSNorm = try_import_RMSNorm()

HEAD_NUM = 32
HIDDEN_SZIE = 4096
SEQ_LEN = 2048
MICRO_BSZ = 1
HEAD_DIM = HIDDEN_SZIE // HEAD_NUM
VOCAB_SIZE = 32000

NUM_KV_HEAD_LIST = [8, 32]
MICRO_BSZ_LIST = [1, 2]
DTYPE_LIST = [torch.bfloat16, torch.float16]
USE_PADDING = [True, False]
VAR_LEN = [True, False]

internlm_accelerator = get_accelerator()


def do_cmp_attn(
    name,
    B,  # pylint: disable=W0613
    S,  # pylint: disable=W0613
    N,
    N_KV,
    q,
    k,
    v,
    dtype,
    attention_mask,  # pylint: disable=W0613
    softmax_scale,
    var_len,
    attention_dropout=0.0,
    cu_seqlens=None,
    **attn_args,  # pylint: disable=W0613
):

    npu_attn_cls = CrossAttention if N != N_KV else SelfAttention
    npu_attn = npu_attn_cls(causal=True, softmax_scale=softmax_scale, attention_dropout=attention_dropout).to(dtype)
    npu_flash_attn = AscendFlashAttention(
        causal=True, softmax_scale=softmax_scale, attention_dropout=attention_dropout
    ).to(dtype)

    q_fa = q.clone()
    k_fa = k.clone()
    v_fa = v.clone()
    if cu_seqlens is not None:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    else:
        max_seqlen = None

    if N == N_KV:
        qkv = torch.concat([q, k, v], dim=2)

        if var_len:
            qkv = unpack_qkv_before_attn(qkv, cu_seqlens)

        a = npu_attn(qkv)  # pylint: disable=E1102

        if var_len:
            a = rearrange(a, "b s h d -> b s (h d)")
            a = pack_output_after_attn(a, cu_seqlens, packed_len=B * S)
    else:
        q = q.squeeze(dim=2)
        kv = torch.concat([k, v], dim=2)
        # import pdb; pdb.set_trace()

        if var_len:
            q = unpack_qkv_before_attn(q, cu_seqlens)
            kv = unpack_qkv_before_attn(kv, cu_seqlens)

        a = npu_attn(q, kv)  # pylint: disable=E1102

        if var_len:
            a = rearrange(a, "b s h d -> b s (h d)")  # recover the shape
            a = pack_output_after_attn(a, cu_seqlens, packed_len=B * S)

    b = npu_flash_attn(  # pylint: disable=E1102
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )

    assert torch.isfinite(a).all().item() and torch.isfinite(b).all().item()

    if dtype == torch.bfloat16:
        # torch_npu's equal not support bfloat16 by now.
        assert torch.allclose(
            a.to(torch.float32), b.to(torch.float32), atol=5e-2, rtol=1e-4
        ), f"{name} not pass, a: {a}, b: {b}"
    else:
        assert torch.allclose(a, b, atol=5e-2, rtol=1e-4), f"{name} not pass, a: {a}, b: {b}"


def npu_transform(B, S, N, N_KV, D, dtype, use_padding, var_len):
    set_random_seed(1024)

    if use_padding:
        x = torch.LongTensor([[i + 1 if i < S // 2 else 0 for i in range(S)] for _ in range(B)]).npu()  # padding S-1024
    else:
        x = torch.LongTensor([[i + 1 for i in range(S)] for _ in range(B)]).npu()  # no-padiing

    cu_seqlens = None
    if var_len:
        cu_seqlens = [0] + sorted(random.sample(list(range(x.numel())), 4))
        if cu_seqlens[-1] != x.numel():
            cu_seqlens.append(x.numel())
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device="npu")

    wq = torch.zeros((N * D, N * D), dtype=dtype, device="npu")
    wk = torch.zeros((N_KV * D, N * D), dtype=dtype, device="npu")
    wv = torch.zeros((N_KV * D, N * D), dtype=dtype, device="npu")
    wembed = torch.zeros((VOCAB_SIZE, HIDDEN_SZIE), dtype=dtype, device="npu")

    # It is very important to set appropriate initialization values for parameters so
    # that the values fall within an appropriate precision range to prevent overflow or underflow.
    with torch.no_grad():
        wq = nn.init.normal_(wq.data)
        wk = nn.init.normal_(wk.data)
        wv = nn.init.normal_(wv.data)
        wembed = nn.init.normal_(wembed.data, std=0.02)

    embed_x = F.embedding(x, wembed).to(dtype)
    q = F.linear(embed_x, wq)  # pylint: disable=E1102
    k = F.linear(embed_x, wk)  # pylint: disable=E1102
    v = F.linear(embed_x, wv)  # pylint: disable=E1102

    if var_len:
        q = rearrange(q, "b s (one h d) -> (b s) one h d", b=B, s=S, d=D, one=1).unsqueeze(0)
        k = rearrange(k, "b s (one h d) -> (b s) one h d", b=B, s=S, d=D, one=1).unsqueeze(0)
        v = rearrange(v, "b s (one h d) -> (b s) one h d", b=B, s=S, d=D, one=1).unsqueeze(0)
    else:
        q = rearrange(q, "b s (one h d) -> b s one h d", b=B, s=S, d=D, one=1)
        k = rearrange(k, "b s (one h d) -> b s one h d", b=B, s=S, d=D, one=1)
        v = rearrange(v, "b s (one h d) -> b s one h d", b=B, s=S, d=D, one=1)

    do_cmp_attn(
        f"B_{B}_S_{S}_N_{N}_N_KV_{N_KV}_D_{D}_{dtype}",
        B,
        S,
        N,
        N_KV,
        q,
        k,
        v,
        dtype,
        None,
        1 / math.sqrt(HIDDEN_SZIE // HEAD_NUM),
        var_len=var_len,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
@pytest.mark.parametrize("test_dtype", DTYPE_LIST)
@pytest.mark.parametrize("num_kv_head", NUM_KV_HEAD_LIST)
@pytest.mark.parametrize("use_padding", USE_PADDING)
@pytest.mark.parametrize("var_len", VAR_LEN)
def test_NPU_fa(micro_bsz, test_dtype, num_kv_head, use_padding, var_len):
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        npu_transform(
            micro_bsz, SEQ_LEN, HEAD_NUM, num_kv_head, HIDDEN_SZIE // HEAD_NUM, test_dtype, use_padding, var_len
        )


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_npu_ops.py"])
