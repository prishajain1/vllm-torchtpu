# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The fused TP32 kernel of the MLA DSpark draft (``kernels.dspark_mla``).

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b.

One call per decode step after the verify, over ``R`` requests with four or
eight verify rows each, on the decoder's 32-rank mesh:

1. **Acceptance.** The verify's vocabulary-sharded logits give each row's
   target (one 31-peer exchange); the accepted count is the leading drafts
   that match, the bonus token (the next anchor) the target after them
   (either may be forced by the caller), as ``kernels.dspark``'s kernel.
2. **Context.** The verify rows' aux features -> ``context_norm(
   context_proj(aux))`` (host-sharded fc and reductions, as there).
3. **Five layers** over the block (the anchor's embedding then mask
   embeddings, at the anchor's position on): RMSNorm, the fused q_a | kv_a
   projection (each rank computes one 128-lane tile and the ranks exchange
   them; the context rows' kv_a for every layer once, before the layers),
   q_a_layernorm,
   this rank's two heads' q_b, RoPE, the absorbed queries; the block and
   context rows' latents; DCP attention (``mla_dcp.DcpAttention``) over the
   draft pools' pages before the verify's first position, with the
   request's new keys (its accepted context rows and its block rows,
   non-causal); W_UV, o_proj (reduce-scattered), the MLP.
4. **Drafts.** The draft's final norm, the target's LM head (vocabulary
   sharded) and exact left-to-right Markov sampling.

Outputs: each request's ``[128]`` int32 vector (``kernels.dspark.OUT_*``
lanes: drafts, targets, count, bonus, margins) and the verify rows'
context latents ``[T, layers, 640]`` for the caller to commit to the draft
pools (``paged.commit_latent_rows``).
"""

from __future__ import annotations

import dataclasses
import operator
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.models.kimi_k3.megakernel.common import collectives32

from . import decode_megakernel as decode_mk
from . import mla_dcp
from .decode_megakernel import PHASE_SCOPES
from .dspark_common import (
    NEG,
    OUT_BONUS,
    OUT_COUNT,
    OUT_DRAFTS,
    OUT_MARGINS,
    OUT_TARGETS,
    SEM_BCAST,
    SEM_GATHER,
    SEM_HOST,
    SEM_RS,
    SEM_TINY,
    SEMS_PER_SLOT,
    pack,
    unpack,
)
from .dspark_common import (
    bf16 as _bf16,
)
from .dspark_common import (
    f32 as _f32,
)
from .dspark_common import (
    rms_norm as _rms_norm,
)
from .dspark_common import (
    select_rows as _select_rows,
)
from .dspark_mla import (
    CHUNK,
    CONTEXT_TILES,
    FC_IN,
    HEADS_PER_RANK,
    LANES,
    MLP_PADDED,
    QKV_TILES,
    RANKS,
    ROWS,
    TILE,
)

_Dynamic = Any
_CONST_16 = 16
_CONST_2 = 2
_CONST_4 = 4
_CONST_32 = 32
_CONST_64 = 64
_CONST_8 = 8


_dot = operator.attrgetter("_dot")(decode_mk)
_phase_end = operator.attrgetter("_phase_end")(decode_mk)
_phase_start = operator.attrgetter("_phase_start")(decode_mk)

LM_CHUNK = 256
LM_CHUNKS = 20  # 20 * 256 = 5120 vocabulary rows per rank
ROW_PACKED = 7168 // 256  # 28 FP32 rows hold one BF16 [1, 7168] row
Q_TILES = QKV_TILES - CONTEXT_TILES  # the q_a tiles; then kv_a's

LAYER_NAMES = ("q_b", "w_uk", "w_uv", "o", "gate", "up", "down")
VMEM_NAMES = (
    "ln1",
    "ln2",
    "q_norm",
    "kv_norm",
    "context_norm",
    "final_norm",
    "mask_embed",
    "rot",
)
HBM_NAMES = (
    "qkv_tile",
    "context_tile",
    "q_b",
    "w_uk",
    "w_uv",
    "o",
    "gate",
    "up",
    "down",
    "fc",
    "w2t",
    "w1",
    "rope",
    "lm_head",
    "embed",
)
WEIGHT_NAMES = VMEM_NAMES + HBM_NAMES
SMALL_NAMES = ("aux", "previous", *VMEM_NAMES)


class _Scalars:
    """An SMEM ref viewed from ``offset``; ``rows`` > 1 maps index ``i`` to.

    entry ``i // rows`` (every row of a request reads its request's).
    """

    def __init__(self, ref: _Dynamic, offset: _Dynamic, rows: _Dynamic = 1) -> None:
        """Execute init logic."""
        self.ref, self.offset, self.rows = ref, offset, rows

    def __getitem__(self, index: _Dynamic) -> _Dynamic:
        """Execute getitem logic."""
        return self.ref[self.offset + (index // self.rows if self.rows > 1 else index)]


def _rope(
    x_bf16: _Dynamic, rot_bf16: _Dynamic, cos32: _Dynamic, sin32: _Dynamic
) -> _Dynamic:
    """Interleaved-pair RoPE in FP32 (``x * cos2 + (x @ R) * sin2``), BF16 out."""
    rotated = _dot(x_bf16, rot_bf16)  # exact: one +-1 term per lane
    return _bf16(_f32(x_bf16) * cos32 + rotated * sin32)


class _DSparkMlaKernel:
    """Pallas kernel body for fused draft MLA."""

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize configuration."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.c,
            self.block,
            self.requests,
            self.rows,
            self.pages,
            self.owner_table,
        ) = (
            _p_next(_p_iter, _p_get("c")),
            _p_next(_p_iter, _p_get("block")),
            _p_next(_p_iter, _p_get("requests")),
            _p_next(_p_iter, _p_get("rows")),
            _p_next(_p_iter, _p_get("pages")),
            _p_next(_p_iter, _p_get("owner_table")),
        )
        self.plan = _p_next(_p_iter, _p_get("plan"))
        self.dp = _p_next(_p_iter, _p_get("dp", 1))

    def __call__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute body logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (scalars, aux_hbm, logits_hbm, previous_hbm) = (
            _p_next(_p_iter, _p_get("scalars")),
            _p_next(_p_iter, _p_get("aux_hbm")),
            _p_next(_p_iter, _p_get("logits_hbm")),
            _p_next(_p_iter, _p_get("previous_hbm")),
        )
        refs = tuple(_p_iter) if len(_p_args) > _CONST_4 else tuple(_p_get("refs", ()))
        runner = _DSparkMlaRunner(
            self, scalars, aux_hbm, logits_hbm, previous_hbm, refs
        )
        return runner.run()


class _DSparkMlaRunner:
    """Execution state and routines for one fused draft MLA invocation."""

    def __getattr__(self, name: _Dynamic) -> _Dynamic:
        """Dynamic attribute access."""
        raise AttributeError(name)

    def __init__(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> None:
        """Initialize execution state."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            self.k,
            self.scalars,
            self.aux_hbm,
            self.logits_hbm,
            self.previous_hbm,
        ) = (
            _p_next(_p_iter, _p_get("kernel")),
            _p_next(_p_iter, _p_get("scalars")),
            _p_next(_p_iter, _p_get("aux_hbm")),
            _p_next(_p_iter, _p_get("logits_hbm")),
            _p_next(_p_iter, _p_get("previous_hbm")),
        )
        refs = _p_next(_p_iter, _p_get("refs", ()))
        kernel = self.k
        self.c = kernel.c
        self.block = kernel.block
        self.r = kernel.requests
        self.rows = kernel.rows
        self.pages = kernel.pages
        self.owner_table = kernel.owner_table
        self.plan = kernel.plan
        self.dp = kernel.dp
        self.t = self.r * self.rows
        self._init_geometry_and_scratch(refs)

    def _init_geometry_and_scratch(self, refs: tuple[_Dynamic, ...]) -> None:
        """Set up rank, dimensions, semaphores, and scratch buffers."""
        c = self.c
        r = self.r
        dp = self.dp
        self.num_layers = c.layers
        self.heads = HEADS_PER_RANK * dp
        self.group_requests, self.group_rows = r // dp, self.t // dp
        self.q_shard = HEADS_PER_RANK * (c.nope + TILE)
        self.vocab = c.vocab // RANKS
        self.hidden = c.hidden
        self.eps = c.eps
        self.scale = c.softmax_scale
        self.drafts_count = self.rows - 1
        self.streamed = self.t > _CONST_32
        self.streamed_o = self.streamed or dp > 1
        self.keys = 2 * self.rows
        self.n_persist = len(mla_dcp.PERSISTENT_NAMES)
        self.n_scoped = len(mla_dcp.SCOPED_NAMES)

        self.sem_fc, self.sem_w2t, self.sem_own, self.sem_lm = 1, 8, 9, 11
        self.sem_embed = 12
        self.sem_rope = 12 + r
        self.sem_w1 = 12 + 3 * r
        self.sem_lat = 12 + 5 * r
        self.sem_down = 13 + 5 * r
        self.sem_tile = 14 + 5 * r
        self.sem_lm2 = 16 + 5 * r
        self.sem_o2, self.sem_down2 = 17 + 5 * r, 18 + 5 * r

        ref_list = list(refs)
        self.weights = dict(
            zip(WEIGHT_NAMES, ref_list[: len(WEIGHT_NAMES)], strict=True)
        )
        ref_list = ref_list[len(WEIGHT_NAMES) :]
        self.pools = tuple(ref_list[: self.num_layers])
        ref_list = ref_list[self.num_layers :]
        self.tokens_out, self.latents_out = ref_list[:2]
        scratch = ref_list[2:]
        self.small = dict(zip(SMALL_NAMES, scratch[: len(SMALL_NAMES)], strict=True))
        scratch = scratch[len(SMALL_NAMES) :]
        self.persistent = scratch[: self.n_persist]
        self.dcp_scoped = scratch[self.n_persist : self.n_persist + self.n_scoped]
        (
            self.q_buf,
            self.w_uk_buf,
            self.w_uv_buf,
            self.o_buf,
            self.gate_buf,
            self.up_buf,
            self.down_buf,
            self.w2t_buf,
            self.w1_blk,
            self.embed_blk,
            self.bcast_buf,
            self.rope_blk,
            self.logits_vmem,
            self.fc_part,
            self.tokens_v,
            self.stage,
            self.local_recv,
            self.host_packed,
            self.gather_recv,
            self.tiny_recv,
            self.argmax_recv,
            self.keys_vmem,
            self.valid_vmem,
            self.latents_vmem,
            self.tile_buf,
            self.tile_recv,
            self.stage2_recv,
            self.stage2_send,
            self.stage2_sems,
            self.dma_sems,
            self.small_sems,
            self.sends,
            self.recvs,
        ) = scratch[self.n_persist + self.n_scoped :]

        self.rank = jax.lax.axis_index("tp")
        self.lane = self.rank % 8
        self.host = self.rank // 8
        self.data_group = self.rank // (RANKS // dp)
        self.pending: dict[str, _Dynamic] = {}
        self.rope_copies: list[_Dynamic] = []
        self.w1_copies: list[list[_Dynamic]] = []

    def own_group(self, value: _Dynamic) -> _Dynamic:
        """This rank's DP group's share of ``value`` (group-major rows)."""
        if self.dp == 1:
            return value
        parts = jnp.split(value, self.dp, axis=0)
        selected = parts[0]
        for g in range(1, self.dp):
            selected = jnp.where(self.data_group == g, parts[g], selected)
        return selected

    def place_group(self, value: _Dynamic) -> _Dynamic:
        """``own_group``'s inverse: the group's rows in place, zeros elsewhere."""
        if self.dp == 1:
            return value
        zeros = jnp.zeros_like(value)
        return jnp.concatenate(
            [jnp.where(self.data_group == g, value, zeros) for g in range(self.dp)],
            axis=0,
        )

    def copy(self, src: _Dynamic, dst: _Dynamic, sem: _Dynamic) -> _Dynamic:
        """Execute copy logic."""
        return tpu.make_async_copy(src, dst, self.dma_sems.at[sem])

    def remote(
        self, src: _Dynamic, dst: _Dynamic, sem_index: _Dynamic, peer: _Dynamic
    ) -> _Dynamic:
        """Execute remote logic."""
        return tpu.make_async_remote_copy(
            src,
            dst,
            self.sends.at[sem_index],
            self.recvs.at[sem_index],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def exchange(
        self,
        src_of: _Dynamic,
        dst_of: _Dynamic,
        offsets: _Dynamic,
        base: _Dynamic,
        slot: _Dynamic,
    ) -> _Dynamic:
        """Execute exchange logic."""
        copies = [
            self.remote(
                src_of(o), dst_of(o), slot * SEMS_PER_SLOT + base + o, self.rank ^ o
            )
            for o in offsets
        ]
        for item in copies:
            item.start()
        for item in copies:
            item.wait()

    def host_reduce_gather(
        self, chunk32: _Dynamic, slot: _Dynamic, tag: _Dynamic = "fc"
    ) -> _Dynamic:
        """FP32 [t, 896] partial of this host lane -> sum over hosts."""
        scope = _phase_start(f"ar_{tag}_host")
        self.host_packed[slot, self.host] = pack(_bf16(chunk32))
        self.exchange(
            lambda _o: self.host_packed.at[slot, self.host],
            lambda _o: self.host_packed.at[slot, self.host],
            (8, 16, 24),
            SEM_HOST,
            slot,
        )
        reduced = sum(
            unpack(self.host_packed[slot, h], self.t, CHUNK) for h in range(4)
        )
        _phase_end(scope)
        scope = _phase_start(f"ar_{tag}_gather")
        self.gather_recv[slot, self.lane] = pack(_bf16(reduced))
        self.exchange(
            lambda _o: self.gather_recv.at[slot, self.lane],
            lambda _o: self.gather_recv.at[slot, self.lane],
            range(1, 8),
            SEM_GATHER,
            slot,
        )
        result = jnp.concatenate(
            [unpack(self.gather_recv[slot, j], self.t, CHUNK) for j in range(8)], axis=1
        )
        _phase_end(scope)
        return result

    def all_reduce(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Lhs @ W summed over the 32 ranks."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            lhs_bf16,
            weight_ref,
            slot,
            refill,
            refill_sems,
            tag,
        ) = (
            _p_next(_p_iter, _p_get("lhs_bf16")),
            _p_next(_p_iter, _p_get("weight_ref")),
            _p_next(_p_iter, _p_get("slot")),
            _p_next(_p_iter, _p_get("refill")),
            _p_next(_p_iter, _p_get("refill_sems", (7, 7))),
            _p_next(_p_iter, _p_get("tag", "")),
        )
        scope = _phase_start(f"ar_{tag}_local")
        copies = []
        own = None
        loading = None
        for k in range(8):
            if refill is not None:
                arrived = loading
                if k + 1 < _CONST_8:
                    loading = self.copy(
                        refill.at[:, pl.ds((k + 1) * CHUNK, CHUNK)],
                        weight_ref.at[(k + 1) % 2],
                        refill_sems[(k + 1) % 2],
                    )
                    loading.start()
                if arrived is not None:
                    arrived.wait()
                operand = weight_ref[k % 2]
            else:
                operand = weight_ref[:, pl.ds(k * CHUNK, CHUNK)]
            partial = _dot(lhs_bf16, operand)
            self.stage[slot, k] = pack(_bf16(partial))
            if k == 0:
                own = self.copy(
                    self.stage.at[slot, 0],
                    self.local_recv.at[slot, self.lane],
                    self.sem_own,
                )
                own.start()
            else:
                item = self.remote(
                    self.stage.at[slot, k],
                    self.local_recv.at[slot, self.lane],
                    slot * SEMS_PER_SLOT + SEM_RS + k,
                    self.rank ^ k,
                )
                item.start()
                copies.append(item)
        _phase_end(scope)
        scope = _phase_start(f"ar_{tag}_rs")
        for item in copies:
            item.wait()
        if not (own is not None):
            _err_msg = "Assertion failed: own is not None"
            raise AssertionError(_err_msg)
        own.wait()
        chunk = sum(unpack(self.local_recv[slot, k], self.t, CHUNK) for k in range(8))
        _phase_end(scope)
        return self.host_reduce_gather(chunk, slot, tag)

    def gather_tiles(
        self, tile_bf16: _Dynamic, slot: _Dynamic, count: _Dynamic
    ) -> _Dynamic:
        """Gather projection tiles from peers."""
        self.tile_recv[slot, self.rank] = pack(tile_bf16)
        copies = [
            self.remote(
                self.tile_recv.at[slot, self.rank],
                self.tile_recv.at[slot, self.rank],
                slot * SEMS_PER_SLOT + SEM_TINY + o,
                self.rank ^ o,
            )
            for o in range(1, 32)
        ]

        @pl.when(self.rank < count)
        def send_tile() -> _Dynamic:
            """Execute send tile logic."""
            for item in copies:
                item.start()

        for o, item in zip(range(1, 32), copies, strict=True):
            pl.when((self.rank ^ o) < count)(item.wait_recv)

        @pl.when(self.rank < count)
        def sent_tile() -> _Dynamic:
            """Execute sent tile logic."""
            for item in copies:
                item.wait_send()

        return [unpack(self.tile_recv[slot, i], self.t, TILE) for i in range(count)]

    def tile_copy(self, source: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute tile copy logic."""
        return self.copy(source, self.tile_buf.at[slot], self.sem_tile + slot)

    def tiny_exchange(self, row_i32: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute tiny exchange logic."""
        self.tiny_recv[slot, self.rank] = row_i32
        self.exchange(
            lambda _o: self.tiny_recv.at[slot, self.rank],
            lambda _o: self.tiny_recv.at[slot, self.rank],
            range(1, 32),
            SEM_TINY,
            slot,
        )
        return self.tiny_recv[slot]

    def start_w1(self, tokens: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute start w1 logic."""
        for j in range(self.r):
            start = pl.multiple_of((tokens[j] // 16) * 16, 16)
            item = self.copy(
                self.weights["w1"].at[pl.ds(start, 16)],
                self.w1_blk.at[slot, j],
                self.sem_w1 + slot * self.r + j,
            )
            item.start()
            self.w1_copies[slot][j] = item

    def w1_rows_for(self, tokens: _Dynamic, slot: _Dynamic) -> _Dynamic:
        """Execute w1 rows for logic."""
        return jnp.concatenate(
            [
                _bf16(_select_rows(_f32(self.w1_blk[slot, j]), tokens[j] % 16, 1))
                for j in range(self.r)
            ],
            axis=0,
        )

    def latent_rows(
        self, kv_bf16: _Dynamic, layer: _Dynamic, cos32: _Dynamic, sin32: _Dynamic
    ) -> _Dynamic:
        """[t, 640] BF16 cache rows: kv_a_layernorm(c_kv) | RoPE(k_pe) | 0."""
        c_kv = _rms_norm(
            kv_bf16[:, : self.c.kv_lora],
            self.small["kv_norm"][layer][:, : self.c.kv_lora],
            self.eps,
        )
        k_pe = _rope(
            kv_bf16[:, self.c.kv_lora : self.c.latent],
            self.small["rot"][...],
            cos32,
            sin32,
        )
        return jnp.concatenate(
            (c_kv, k_pe, jnp.zeros((self.t, LANES - self.c.latent), jnp.bfloat16)),
            axis=1,
        )

    def layer_copy(self, name: _Dynamic, layer: _Dynamic) -> _Dynamic:
        """Execute layer copy logic."""
        buffers = {
            "q_b": (self.weights["q_b"].at[layer], self.q_buf, 2),
            "w_uk": (self.weights["w_uk"].at[layer], self.w_uk_buf, 3),
            "w_uv": (self.weights["w_uv"].at[layer], self.w_uv_buf, 4),
            "o": (
                self.weights["o"].at[layer, :, pl.ds(0, self.o_buf.shape[-1])],
                self.o_buf.at[0] if self.streamed_o else self.o_buf,
                5,
            ),
            "gate": (self.weights["gate"].at[layer], self.gate_buf, 6),
            "up": (self.weights["up"].at[layer], self.up_buf, 7),
            "down": (
                self.weights["down"].at[layer, :, pl.ds(0, self.down_buf.shape[-1])],
                self.down_buf.at[0] if self.streamed else self.down_buf,
                self.sem_down,
            ),
        }
        item = self.copy(*buffers[name])
        item.start()
        return item

    def refill(self, name: _Dynamic, layer: _Dynamic) -> _Dynamic:
        """Execute refill logic."""
        if layer + 1 < self.num_layers:
            self.pending[name] = self.layer_copy(name, layer + 1)

    def _run_layer_qkv(
        self,
        layer: _Dynamic,
        x: _Dynamic,
        tile_copies: list[_Dynamic],
        cos_block: _Dynamic,
        sin_block: _Dynamic,
    ) -> _Dynamic:
        """Run QKV projection and key updates for one draft layer."""
        scope = _phase_start("d_qkv")
        h = _rms_norm(x, self.small["ln1"][layer], self.eps)
        slot = layer % 2
        tile_copies[slot].wait()
        tiles = self.gather_tiles(
            _bf16(_dot(h, self.tile_buf[slot])), (layer + 1) % 2, QKV_TILES
        )
        if layer + 2 < self.num_layers:
            tile_copies[slot] = self.tile_copy(
                self.weights["qkv_tile"].at[layer + 2], slot
            )
            tile_copies[slot].start()
        q_a = _bf16(jnp.concatenate(tiles[:Q_TILES], axis=1))
        block_kv = _bf16(jnp.concatenate(tiles[Q_TILES:], axis=1))
        q_c = self.own_group(_rms_norm(q_a, self.small["q_norm"][layer], self.eps))
        block_latents = self.own_group(
            self.latent_rows(block_kv, layer, cos_block, sin_block)
        )
        context_latents = self.own_group(self.latents_vmem[layer])
        for j in range(self.group_requests):
            self.keys_vmem[j] = jnp.concatenate(
                (
                    _f32(context_latents[j * self.rows : (j + 1) * self.rows]),
                    _f32(block_latents[j * self.rows : (j + 1) * self.rows]),
                ),
                axis=0,
            )
        _phase_end(scope)
        return q_c

    def _run_layer_query(
        self,
        layer: _Dynamic,
        q_c: _Dynamic,
        cos_query: _Dynamic,
        sin_query: _Dynamic,
    ) -> _Dynamic:
        """Run query projection and RoPE."""
        scope = _phase_start("d_query")
        self.pending["q_b"].wait()
        q = _bf16(_dot(q_c, self.q_buf[...]))
        self.refill("q_b", layer)
        self.pending["w_uk"].wait()
        queries = []
        for head in range(self.heads):
            shard, within = divmod(head, HEADS_PER_RANK)
            nope = shard * self.q_shard + within * self.c.nope
            q_abs = _bf16(_dot(q[:, nope : nope + self.c.nope], self.w_uk_buf[head]))
            pe = shard * self.q_shard + HEADS_PER_RANK * self.c.nope + within * 128
            q_pe = _rope(
                q[:, pe : pe + self.c.rope],
                self.small["rot"][...],
                cos_query,
                sin_query,
            )
            queries.append(
                jnp.concatenate(
                    (
                        _f32(q_abs) * self.scale,
                        _f32(q_pe) * self.scale,
                        jnp.zeros(
                            (self.group_rows, LANES - self.c.latent), jnp.float32
                        ),
                    ),
                    axis=1,
                )
            )
        queries = jnp.stack(queries, axis=1)
        self.refill("w_uk", layer)
        _phase_end(scope)
        return queries

    def _run_layer_attend(
        self,
        layer: _Dynamic,
        queries: _Dynamic,
        first_request: _Dynamic,
        cache: _Dynamic,
        workspace: _Dynamic,
    ) -> _Dynamic:
        """Run attention via DCP."""
        scope = _phase_start("d_attend")
        attention = mla_dcp.DcpAttention(
            dataclasses.replace(cache, layer=jnp.int32(layer)),
            workspace,
            _Scalars(self.scalars, first_request, self.rows),
            self.group_rows,
            sequence_rows=self.rows,
            heads=self.heads,
            chunk_rows=self.rows,
        )
        attention.lse_in_padding = self.block < self.rows
        attention.group_fold = True
        if self.block < self.rows and mla_dcp.GROUP == RANKS:
            attention.force_chunked = True
            attention.host_combine = (
                self.stage2_recv,
                self.stage2_send,
                self.stage2_sems,
            )
        if PHASE_SCOPES:
            attention.trace = "a_"
        attention.prime()
        merged = attention.attend(
            queries,
            jnp.zeros((self.group_rows, LANES), jnp.float32),
            layer % 2,
            keys=self.keys_vmem,
            key_valid=self.valid_vmem,
        )
        _phase_end(scope)
        return merged

    def _run_layer_mlp(
        self, layer: _Dynamic, x: _Dynamic, merged: _Dynamic
    ) -> _Dynamic:
        """Run output projection and MLP."""
        scope = _phase_start("d_o")
        wuv_scope = _phase_start("d_wuv")
        self.pending["w_uv"].wait()
        attended = self.place_group(
            jnp.concatenate(
                [
                    _bf16(_dot(_bf16(merged[:, head]), self.w_uv_buf[head]))
                    for head in range(self.heads)
                ],
                axis=1,
            )
        )
        self.refill("w_uv", layer)
        self.pending["o"].wait()
        _phase_end(wuv_scope)
        x = _bf16(
            _f32(x)
            + self.all_reduce(
                attended,
                self.o_buf,
                1,
                refill=self.weights["o"].at[layer] if self.streamed_o else None,
                refill_sems=(5, self.sem_o2),
                tag="o",
            )
        )
        self.refill("o", layer)
        _phase_end(scope)
        scope = _phase_start("d_mlp")
        h2 = _rms_norm(x, self.small["ln2"][layer], self.eps)
        self.pending["gate"].wait()
        self.pending["up"].wait()
        gate = _f32(_bf16(_dot(h2, self.gate_buf[...])))
        up = _bf16(_dot(h2, self.up_buf[...]))
        self.refill("gate", layer)
        self.refill("up", layer)
        activation = _bf16(_f32(_bf16(gate * jax.nn.sigmoid(gate))) * _f32(up))
        _phase_end(scope)
        scope = _phase_start("d_down")
        self.pending["down"].wait()
        mlp = self.all_reduce(
            activation,
            self.down_buf,
            0,
            refill=self.weights["down"].at[layer] if self.streamed else None,
            refill_sems=(self.sem_down, self.sem_down2),
            tag="down",
        )
        self.refill("down", layer)
        x = _bf16(_f32(x) + mlp)
        _phase_end(scope)
        return x

    def run_layer(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Run attention and MLP for one draft layer."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            layer,
            x,
            tile_copies,
            cos_block,
            sin_block,
            first_request,
            cache,
            workspace,
            cos_query,
            sin_query,
        ) = (
            _p_next(_p_iter, _p_get("layer")),
            _p_next(_p_iter, _p_get("x")),
            _p_next(_p_iter, _p_get("tile_copies")),
            _p_next(_p_iter, _p_get("cos_block")),
            _p_next(_p_iter, _p_get("sin_block")),
            _p_next(_p_iter, _p_get("first_request")),
            _p_next(_p_iter, _p_get("cache")),
            _p_next(_p_iter, _p_get("workspace")),
            _p_next(_p_iter, _p_get("cos_query")),
            _p_next(_p_iter, _p_get("sin_query")),
        )
        q_c = self._run_layer_qkv(layer, x, tile_copies, cos_block, sin_block)
        queries = self._run_layer_query(layer, q_c, cos_query, sin_query)
        merged = self._run_layer_attend(layer, queries, first_request, cache, workspace)
        return self._run_layer_mlp(layer, x, merged)

    def _run_prologue(self) -> tuple[_Dynamic, ...]:
        """Set up positions, barriers, and kick off asynchronous copies."""
        r = self.r
        new_position = jnp.stack([self.scalars[j] for j in range(r)])
        forced_count = jnp.stack([self.scalars[r + j] for j in range(r)])
        forced_anchor = jnp.stack([self.scalars[2 * r + j] for j in range(r)])

        collectives32.barrier()
        sources = {
            "aux": self.aux_hbm,
            "previous": self.previous_hbm,
            **{n: self.weights[n] for n in VMEM_NAMES},
        }
        small_copies = [
            tpu.make_async_copy(sources[n], self.small[n], self.small_sems.at[i])
            for i, n in enumerate(SMALL_NAMES)
        ]
        for item in small_copies:
            item.start()
        logits_copy = self.copy(self.logits_hbm, self.logits_vmem, 0)
        logits_copy.start()
        w2t_copy = self.copy(self.weights["w2t"], self.w2t_buf, self.sem_w2t)
        w2t_copy.start()
        tile_copies = [
            self.tile_copy(self.weights["qkv_tile"].at[0], 0),
            self.tile_copy(self.weights["context_tile"], 1),
        ]
        for item in tile_copies:
            item.start()
        return (
            new_position,
            forced_count,
            forced_anchor,
            small_copies,
            logits_copy,
            w2t_copy,
            tile_copies,
        )

    def _send_anchor_row(self, j: int, anchor_token: _Dynamic, owner: _Dynamic) -> None:
        """Send anchor row across ranks if current rank is the owner."""

        @pl.when(self.rank == owner[j])
        def _send() -> None:
            """Send anchor row across ranks."""
            local_id = anchor_token[j] - owner[j] * self.vocab
            item = self.copy(
                self.weights["embed"].at[
                    pl.ds(pl.multiple_of((local_id // 16) * 16, 16), 16)
                ],
                self.embed_blk,
                self.sem_embed + j,
            )
            item.start()
            item.wait()
            self.bcast_buf[j] = pack(
                _bf16(_select_rows(_f32(self.embed_blk[...]), local_id % 16, 1))
            )
            copies = [
                self.remote(
                    self.bcast_buf.at[j],
                    self.bcast_buf.at[j],
                    SEM_BCAST + j,
                    self.rank ^ o,
                )
                for o in range(1, 32)
            ]
            for c_item in copies:
                c_item.start()
            for c_item in copies:
                c_item.wait_send()

    def _run_gather_argmax(
        self,
        small_copies: list[_Dynamic],
        logits_copy: _Dynamic,
    ) -> tuple[_Dynamic, _Dynamic]:
        """Gather argmax across ranks and calculate targets and margins."""
        r, rows, t = self.r, self.rows, self.t
        for item in small_copies:
            item.wait()
        logits_copy.wait()
        verify_logits = self.logits_vmem[...]
        vocab_iota = jax.lax.broadcasted_iota(jnp.int32, (t, self.vocab), 1)
        local_best = jnp.max(verify_logits, axis=1, keepdims=True)
        local_id = jnp.min(
            jnp.where(verify_logits == local_best, vocab_iota, self.vocab),
            axis=1,
            keepdims=True,
        )
        local_second = jnp.max(
            jnp.where(vocab_iota == local_id, NEG, verify_logits), axis=1, keepdims=True
        )
        out_lane = jax.lax.broadcasted_iota(jnp.int32, (t, 128), 1)
        packet = jnp.where(
            out_lane == 0,
            jax.lax.bitcast_convert_type(local_best, jnp.int32),
            jnp.where(
                out_lane == 1,
                local_id + self.rank * self.vocab,
                jnp.where(
                    out_lane == _CONST_2,
                    jax.lax.bitcast_convert_type(local_second, jnp.int32),
                    0,
                ),
            ),
        )
        square = jnp.pad(packet, ((0, 128 - t), (0, 0)))
        self.argmax_recv[self.rank] = jnp.transpose(square)[:8]
        self.exchange(
            lambda _o: self.argmax_recv.at[self.rank],
            lambda _o: self.argmax_recv.at[self.rank],
            range(1, 32),
            SEM_TINY,
            1,
        )
        gathered = self.argmax_recv[...]
        rank_iota3 = jax.lax.broadcasted_iota(jnp.int32, (RANKS, 1, 128), 0)
        bests = jax.lax.bitcast_convert_type(gathered[:, 0:1, :], jnp.float32)
        seconds = jax.lax.bitcast_convert_type(gathered[:, 2:3, :], jnp.float32)
        best = jnp.max(bests, axis=0)
        winner = jnp.min(jnp.where(bests == best[None], rank_iota3, RANKS), axis=0)
        target_lanes = jnp.sum(
            jnp.where(rank_iota3 == winner[None], gathered[:, 1:2, :], 0), axis=0
        )
        runner_up = jnp.maximum(
            jnp.max(seconds, axis=0),
            jnp.max(jnp.where(rank_iota3 == winner[None], NEG, bests), axis=0),
        )
        lanes_back = jnp.concatenate(
            (
                target_lanes,
                jax.lax.bitcast_convert_type(best - runner_up, jnp.int32),
                jnp.zeros((126, 128), jnp.int32),
            ),
            axis=0,
        )
        rows_back = jnp.transpose(lanes_back)[:t]
        targets = rows_back[:, 0:1].reshape(r, rows, 1)
        margins = jax.lax.bitcast_convert_type(rows_back[:, 1:2], jnp.float32).reshape(
            r, rows, 1
        )
        return targets, margins

    def _run_anchor_and_ropes(
        self,
        new_position: _Dynamic,
        count: _Dynamic,
        anchor_token: _Dynamic,
        targets: _Dynamic,
        margins: _Dynamic,
    ) -> _Dynamic:
        """Fill token outputs and kick off RoPE and embedding broadcasts."""
        r, rows = self.r, self.rows
        anchor_position = new_position + count + 1
        result_lane = jax.lax.broadcasted_iota(jnp.int32, (r, 128), 1)
        field_row = jax.lax.broadcasted_iota(jnp.int32, (r, rows, 128), 1)
        field_lane = jax.lax.broadcasted_iota(jnp.int32, (r, rows, 128), 2)
        targets_row = jnp.sum(
            jnp.where(field_lane == field_row + OUT_TARGETS, targets, 0), axis=1
        )
        margins_row = jnp.sum(
            jnp.where(
                field_lane == field_row + OUT_MARGINS,
                jax.lax.bitcast_convert_type(margins, jnp.int32),
                0,
            ),
            axis=1,
        )
        result = jnp.zeros((r, 128), jnp.int32) - 1
        result = jnp.where(
            (result_lane >= OUT_TARGETS) & (result_lane < OUT_TARGETS + rows),
            targets_row,
            result,
        )
        result = jnp.where(
            (result_lane >= OUT_MARGINS) & (result_lane < OUT_MARGINS + rows),
            margins_row,
            result,
        )
        result = jnp.where(result_lane == OUT_COUNT, count[:, None], result)
        result = jnp.where(result_lane == OUT_BONUS, anchor_token[:, None], result)
        self.tokens_v[...] = result

        key_lane = jax.lax.broadcasted_iota(jnp.int32, (r, 1, self.keys), 2)
        self.valid_vmem[...] = self.own_group(
            (
                ((key_lane < rows) & (key_lane <= count[:, None, None]))
                | ((key_lane >= rows) & (key_lane < rows + self.block))
            ).astype(jnp.int32)
        )

        self.w1_copies = [[None] * r, [None] * r]
        self.start_w1(anchor_token, 0)
        rope_copies = []
        for j in range(r):
            block_start = pl.multiple_of((anchor_position[j] // 8) * 8, 8)
            context_start = pl.multiple_of((new_position[j] // 8) * 8, 8)
            rope_copies += (
                self.copy(
                    self.weights["rope"].at[pl.ds(block_start, 16)],
                    self.rope_blk.at[0, j],
                    self.sem_rope + 2 * j,
                ),
                self.copy(
                    self.weights["rope"].at[pl.ds(context_start, 16)],
                    self.rope_blk.at[1, j],
                    self.sem_rope + 2 * j + 1,
                ),
            )
        for item in rope_copies:
            item.start()
        self.rope_copies = rope_copies

        owner = anchor_token // self.vocab
        for j in range(r):
            self._send_anchor_row(j, anchor_token, owner)
        return anchor_position

    def _run_verify_and_anchor(
        self,
        new_position: _Dynamic,
        forced_count: _Dynamic,
        forced_anchor: _Dynamic,
        small_copies: list[_Dynamic],
        logits_copy: _Dynamic,
    ) -> tuple[_Dynamic, _Dynamic]:
        """Verify drafts and select anchor token."""
        r, rows = self.r, self.rows
        scope = _phase_start("d_accept")
        targets, margins = self._run_gather_argmax(small_copies, logits_copy)
        previous_vector = self.small["previous"][...]
        row_iota_col = jax.lax.broadcasted_iota(jnp.int32, (r, rows, 1), 1)
        proposals = jnp.sum(
            jnp.where(
                jax.lax.broadcasted_iota(jnp.int32, (r, rows, 128), 2)
                == row_iota_col + OUT_DRAFTS,
                jnp.broadcast_to(previous_vector[:, None, :], (r, rows, 128)),
                0,
            ),
            axis=2,
            keepdims=True,
        )
        match_rows = (row_iota_col < self.drafts_count) & (proposals == targets)
        first_miss = jnp.min(
            jnp.where(
                match_rows | (row_iota_col >= self.drafts_count), rows, row_iota_col
            ),
            axis=1,
        )[:, 0]
        count = jnp.where(
            forced_count >= 0, forced_count, jnp.minimum(first_miss, self.drafts_count)
        )
        anchor_token = jnp.sum(
            jnp.where(row_iota_col[:, :, 0] == count[:, None], targets[:, :, 0], 0),
            axis=1,
        )
        anchor_token = jnp.where(forced_anchor >= 0, forced_anchor, anchor_token)
        anchor_position = self._run_anchor_and_ropes(
            new_position, count, anchor_token, targets, margins
        )
        _phase_end(scope)
        return anchor_token, anchor_position

    def _build_fc_spans(self) -> list[list[tuple[int, int, int, int]]]:
        """Build per-host FC auxiliary layer slice spans."""
        spans: list[list[tuple[int, int, int, int]]] = []
        for h in range(4):
            start, stop = h * FC_IN, (h + 1) * FC_IN
            pieces = []
            for layer_index in range(len(self.c.aux_layers)):
                lo, hi = layer_index * self.hidden, (layer_index + 1) * self.hidden
                a, b = max(start, lo), min(stop, hi)
                if a < b:
                    pieces.append((layer_index, a - lo, b - lo, a - start))
            spans.append(pieces)
        return spans

    def _run_context_phase(self) -> _Dynamic:
        """Compute FC context projection and RMSNorm."""
        scope = _phase_start("d_context")
        aux_ref = self.small["aux"]
        spans = self._build_fc_spans()

        fc_parts = 2
        part_rows = FC_IN // fc_parts

        def fc_phase(fc_buf: _Dynamic) -> None:
            """Stream FC weights in two halves and accumulate host partials."""
            for p in range(fc_parts):
                item = self.copy(
                    self.weights["fc"].at[pl.ds(p * part_rows, part_rows)],
                    fc_buf,
                    self.sem_fc,
                )
                item.start()
                item.wait()
                for h in range(4):

                    @pl.when(self.host == h)
                    def fc_branch(h: int = h, p: int = p) -> None:
                        """Accumulate host partial for host h and part p."""
                        total = None if p == 0 else self.fc_part[...]
                        for layer_index, lo, hi, offset in spans[h]:
                            a, b = (
                                max(offset, p * part_rows),
                                min(offset + hi - lo, (p + 1) * part_rows),
                            )
                            if a >= b:
                                continue
                            part = _dot(
                                aux_ref[
                                    layer_index, :, lo + a - offset : lo + b - offset
                                ],
                                fc_buf[pl.ds(a - p * part_rows, b - a), :],
                            )
                            total = part if total is None else total + part
                        if total is not None:
                            self.fc_part[...] = total

        pl.run_scoped(fc_phase, tpu.VMEM((part_rows, CHUNK), jnp.bfloat16))
        context = _rms_norm(
            _bf16(self.host_reduce_gather(self.fc_part[...], 0)),
            self.small["context_norm"][...],
            self.eps,
        )
        _phase_end(scope)
        return context

    def _run_context_and_input(
        self,
        new_position: _Dynamic,
        anchor_token: _Dynamic,
        anchor_position: _Dynamic,
        tile_copies: list[_Dynamic],
    ) -> tuple[_Dynamic, ...]:
        """Compute context features and initial input embeddings."""
        r, rows, t = self.r, self.rows, self.t
        context = self._run_context_phase()

        scope = _phase_start("d_input")
        for item in self.rope_copies:
            item.wait()
        rope_block = jnp.concatenate(
            [
                _select_rows(self.rope_blk[0, j], anchor_position[j] % 8, rows)
                for j in range(r)
            ],
            axis=0,
        )
        rope_context = jnp.concatenate(
            [
                _select_rows(self.rope_blk[1, j], new_position[j] % 8, rows)
                for j in range(r)
            ],
            axis=0,
        )
        cos_block, sin_block = (
            rope_block[:, : self.c.rope],
            rope_block[:, self.c.rope :],
        )
        cos_context, sin_context = (
            rope_context[:, : self.c.rope],
            rope_context[:, self.c.rope :],
        )
        owner = anchor_token // self.vocab
        for j in range(r):
            pl.when(self.rank != owner[j])(
                self.remote(
                    self.bcast_buf.at[j],
                    self.bcast_buf.at[j],
                    SEM_BCAST + j,
                    owner[j],
                ).wait_recv
            )

        anchor_rows = jnp.concatenate(
            [unpack(self.bcast_buf[j], 1, self.hidden) for j in range(r)], axis=0
        )
        mask_row = _f32(self.small["mask_embed"][...])
        x_row = jax.lax.broadcasted_iota(jnp.int32, (r, rows, self.hidden), 1)
        x = _bf16(
            jnp.where(x_row == 0, anchor_rows[:, None, :], mask_row[:, None, :])
        ).reshape(t, self.hidden)

        tile_copies[1].wait()
        context_tiles = self.gather_tiles(
            _bf16(_dot(context, self.tile_buf[1])), 0, self.num_layers * CONTEXT_TILES
        )
        if self.num_layers > 1:
            tile_copies[1] = self.tile_copy(self.weights["qkv_tile"].at[1], 1)
            tile_copies[1].start()
        for layer in range(self.num_layers):
            context_kv = _bf16(
                jnp.concatenate(
                    context_tiles[layer * CONTEXT_TILES : (layer + 1) * CONTEXT_TILES],
                    axis=1,
                )
            )
            self.latents_vmem[layer] = self.latent_rows(
                context_kv, layer, cos_context, sin_context
            )

        first_request = self.data_group * self.group_requests if self.dp > 1 else 0
        page_owner = None
        if self.owner_table is not None:
            page_owner = mla_dcp.owner_index(self.owner_table)
            page_owner = page_owner % mla_dcp.GROUP if self.dp > 1 else page_owner
        cache = mla_dcp.DcpCache(
            pool_ref=self.pools,
            table_ref=_Scalars(self.scalars, 3 * r),
            pages=self.pages,
            layer=jnp.int32(0),
            group=mla_dcp.GROUP,
            owner=page_owner,
            table_row=lambda row: first_request + row // rows,
        )
        cos_query, sin_query = self.own_group(cos_block), self.own_group(sin_block)
        workspace = mla_dcp.DcpWorkspace(*self.persistent, *self.dcp_scoped)

        for name in LAYER_NAMES:
            self.pending[name] = self.layer_copy(name, 0)
        _phase_end(scope)

        return (
            x,
            tile_copies,
            cos_block,
            sin_block,
            first_request,
            cache,
            workspace,
            cos_query,
            sin_query,
        )

    def _run_dcp_layers(self, *_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Run attention and MLP across all draft layers."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            x,
            tile_copies,
            cos_block,
            sin_block,
            first_request,
            cache,
            workspace,
            cos_query,
            sin_query,
        ) = (
            _p_next(_p_iter, _p_get("x")),
            _p_next(_p_iter, _p_get("tile_copies")),
            _p_next(_p_iter, _p_get("cos_block")),
            _p_next(_p_iter, _p_get("sin_block")),
            _p_next(_p_iter, _p_get("first_request")),
            _p_next(_p_iter, _p_get("cache")),
            _p_next(_p_iter, _p_get("workspace")),
            _p_next(_p_iter, _p_get("cos_query")),
            _p_next(_p_iter, _p_get("sin_query")),
        )
        for layer in range(self.num_layers):
            x = self.run_layer(
                layer,
                x,
                tile_copies,
                cos_block,
                sin_block,
                first_request,
                cache,
                workspace,
                cos_query,
                sin_query,
            )
        return x

    def _run_lm_head_and_markov(
        self,
        x: _Dynamic,
        anchor_token: _Dynamic,
        w2t_copy: _Dynamic,
    ) -> None:
        """Compute LM head and Markov draft predictions."""
        r, rows = self.r, self.rows
        scope = _phase_start("d_lm")
        latents_copy = self.copy(self.latents_vmem, self.latents_out, self.sem_lat)
        latents_copy.start()
        hidden_rows = _rms_norm(x, self.small["final_norm"][...], self.eps)

        def lm_phase(chunk_a: _Dynamic, chunk_b: _Dynamic) -> None:
            """Compute vocabulary-sharded LM head logits in chunks."""
            buffers = (chunk_a, chunk_b)

            def chunk_copy(k: int) -> _Dynamic:
                """Start async copy of LM head weight chunk k."""
                item = self.copy(
                    self.weights["lm_head"].at[:, pl.ds(k * LM_CHUNK, LM_CHUNK)],
                    buffers[k % 2],
                    (self.sem_lm, self.sem_lm2)[k % 2],
                )
                item.start()
                return item

            copies = [chunk_copy(0), chunk_copy(1)]
            for k in range(LM_CHUNKS):
                copies[k % 2].wait()
                logits = _f32(_bf16(_dot(hidden_rows, buffers[k % 2][...])))
                if k + 2 < LM_CHUNKS:
                    copies[k % 2] = chunk_copy(k + 2)
                self.logits_vmem[:, pl.ds(k * LM_CHUNK, LM_CHUNK)] = logits

        pl.run_scoped(
            lm_phase,
            tpu.VMEM((self.hidden, LM_CHUNK), jnp.bfloat16),
            tpu.VMEM((self.hidden, LM_CHUNK), jnp.bfloat16),
        )
        _phase_end(scope)

        scope = _phase_start("d_markov")
        w2t_copy.wait()
        vocab_lane = jax.lax.broadcasted_iota(jnp.int32, (1, self.vocab), 1)
        tokens = self.tokens_v[...]
        token_lane = jax.lax.broadcasted_iota(jnp.int32, (r, 128), 1)
        rank_iota = jax.lax.broadcasted_iota(jnp.int32, (RANKS, r, 128), 0)
        previous = anchor_token
        for slot in range(self.block):
            for item in self.w1_copies[slot % 2]:
                item.wait()
            bias = _dot(self.w1_rows_for(previous, slot % 2), self.w2t_buf[...])
            scores = (
                jnp.concatenate(
                    [
                        self.logits_vmem[j * rows + slot : j * rows + slot + 1, :]
                        for j in range(r)
                    ],
                    axis=0,
                )
                + bias
            )
            best_value = jnp.max(scores, axis=1, keepdims=True)
            best_id = jnp.min(
                jnp.where(scores == best_value, vocab_lane, self.vocab),
                axis=1,
                keepdims=True,
            )
            row = jnp.where(
                token_lane == 0,
                jax.lax.bitcast_convert_type(best_value, jnp.int32),
                jnp.where(token_lane == 1, best_id + self.rank * self.vocab, 0),
            )
            gathered = self.tiny_exchange(row, slot % 2)
            maxima = jax.lax.bitcast_convert_type(gathered[:, :, 0:1], jnp.float32)
            top = jnp.max(maxima, axis=0, keepdims=True)
            win = jnp.min(
                jnp.where(maxima == top, rank_iota[:, :, 0:1], RANKS),
                axis=0,
                keepdims=True,
            )
            token = jnp.sum(jnp.where(rank_iota == win, gathered, 0), axis=0)[:, 1:2]
            tokens = jnp.where(token_lane == OUT_DRAFTS + slot, token, tokens)
            previous = token[:, 0]
            if slot + 1 < self.block:
                self.start_w1(previous, (slot + 1) % 2)
        self.tokens_v[...] = tokens
        out = tpu.make_async_copy(self.tokens_v, self.tokens_out, self.small_sems.at[0])
        out.start()
        out.wait()
        latents_copy.wait()
        _phase_end(scope)

    def run(self) -> _Dynamic:
        """Run the full fused draft MLA kernel."""
        (
            new_position,
            forced_count,
            forced_anchor,
            small_copies,
            logits_copy,
            w2t_copy,
            tile_copies,
        ) = self._run_prologue()
        anchor_token, anchor_position = self._run_verify_and_anchor(
            new_position,
            forced_count,
            forced_anchor,
            small_copies,
            logits_copy,
        )
        (
            x,
            tile_copies,
            cos_block,
            sin_block,
            first_request,
            cache,
            workspace,
            cos_query,
            sin_query,
        ) = self._run_context_and_input(
            new_position,
            anchor_token,
            anchor_position,
            tile_copies,
        )
        x = self._run_dcp_layers(
            x,
            tile_copies,
            cos_block,
            sin_block,
            first_request,
            cache,
            workspace,
            cos_query,
            sin_query,
        )
        self._run_lm_head_and_markov(x, anchor_token, w2t_copy)
        return None


def _make_body(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """Create Pallas kernel body for fused draft MLA."""
    return _DSparkMlaKernel(*_p_args, **_p_kw)


def fused_draft_local(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
    """The per-shard fused draft (call inside a TP32 ``shard_map``):.

    ``(weights, pools, aux, logits, previous, positions0, forced_counts,
    forced_anchors, tables)`` -> ``(vectors [r, 128] int32, latents [layers,
    t, 640] bf16)``.
    Under ``attention_dp`` (DCP over each group, ``mla_dcp.GROUP``) the first
    ``r / dp`` requests are group 0's, and so on: a rank attends its group's
    requests with its held TP32 shards' heads (``q_b``, ``w_uk``, ``w_uv``
    and ``o`` hold ``dp`` shards, ``dspark_mla.held_shards``).
    ``weights``: ``WEIGHT_NAMES`` (this rank's arrays, including the target's
    ``lm_head`` ``[7168, 5120]`` and ``embed`` ``[5120, 7168]``); ``pools``:
    the draft layers' local pools ``[blocks, 128, packing, 640]``; ``aux``
    ``[aux layers, t, hidden]`` and ``logits`` ``[r, rows, vocab / 32]`` the
    verify's (``t = rows * r``; four or eight rows per request); ``previous``
    ``[r, 128]`` the last vectors; ``positions0`` ``[r]`` each verify's first position;
    ``forced_*`` ``[r]`` (``>= 0``: given); ``tables`` ``[r, pages]``.
    """
    _p_iter = iter(_p_args)
    _p_get: _Dynamic = _p_kw.get
    _p_next: _Dynamic = next
    c = _p_next(_p_iter, _p_get("c"))
    (
        requests,
        pages,
        block,
        rows,
        owner_table,
        attention_dp,
    ) = (
        _p_get("requests"),
        _p_get("pages"),
        _p_get("block"),
        _p_get("rows", ROWS),
        _p_get("owner_table"),
        _p_get("attention_dp", 1),
    )
    block = c.block if block is None else block
    if rows not in (4, ROWS):
        _err_msg = "the fused MLA draft takes 4 or 8 rows per request"
        raise ValueError(_err_msg)
    if not _CONST_2 <= block <= rows:
        _err_msg = f"block must be 2..{rows}"
        raise ValueError(_err_msg)
    r = requests
    t = r * rows
    if not 1 <= r <= _CONST_16 or t > _CONST_64:
        _err_msg = (
            "the fused MLA draft takes up to 8 requests at 8 rows, or 16 at 4 rows"
        )
        raise ValueError(_err_msg)
    dp = attention_dp
    if r % dp or mla_dcp.GROUP * dp != RANKS:
        _err_msg = f"attention DP{dp}: {r} requests over DCP{mla_dcp.GROUP} groups"
        raise ValueError(_err_msg)
    heads, group_rows = HEADS_PER_RANK * dp, t // dp
    packed = t * CHUNK // 256
    streamed = t > _CONST_32

    def call(*_p_args: _Dynamic, **_p_kw: _Dynamic) -> _Dynamic:
        """Execute call logic."""
        _p_iter = iter(_p_args)
        _p_get: _Dynamic = _p_kw.get
        _p_next: _Dynamic = next
        (
            weights,
            pools,
            aux,
            logits,
            previous,
            position,
            forced_count,
            forced_anchor,
            tables,
        ) = (
            _p_next(_p_iter, _p_get("weights")),
            _p_next(_p_iter, _p_get("pools")),
            _p_next(_p_iter, _p_get("aux")),
            _p_next(_p_iter, _p_get("logits")),
            _p_next(_p_iter, _p_get("previous")),
            _p_next(_p_iter, _p_get("position")),
            _p_next(_p_iter, _p_get("forced_count")),
            _p_next(_p_iter, _p_get("forced_anchor")),
            _p_next(_p_iter, _p_get("tables")),
        )
        pools = tuple(pools)
        logits = logits.reshape(t, c.vocab // RANKS).astype(jnp.float32)
        cache_dtype = pools[0].dtype
        page_tokens = pools[0].shape[1] * pools[0].shape[2]
        chunks = (
            block < rows and mla_dcp.GROUP == RANKS
        )  # the two-stage combine (see the body)
        plan = mla_dcp.page_plan(
            group_rows, heads, rows, cache_dtype, force_chunks=chunks
        )
        scalars = jnp.concatenate(
            (position, forced_count, forced_anchor, tables.reshape(-1))
        ).astype(jnp.int32)
        small_shapes = {"aux": aux, "previous": previous, **weights}
        # FP8 caches: FP8 query packets (the operand the receiver would
        # quantize to anyway), half the gather's bytes.
        persistent = mla_dcp.persistent_scratch(
            group_rows,
            mla_dcp.GROUP,
            heads,
            cache_dtype=cache_dtype,
            sequence_rows=rows,
            fp8_wire=True,
            force_chunks=chunks,
            chunk_rows=rows,
        )
        ring = persistent[0].shape[0] if mla_dcp.chunked(group_rows, chunks) else 1
        hosts = RANKS // mla_dcp.HOST_LANES
        scratch = (
            *(
                tpu.VMEM(small_shapes[n].shape, small_shapes[n].dtype)
                for n in SMALL_NAMES
            ),
            *persistent,
            *mla_dcp.scoped_scratch(
                group_rows,
                plan["span"],
                plan["buffers"],
                heads=heads,
                group=mla_dcp.GROUP,
                cache_dtype=cache_dtype,
                page_tokens=page_tokens,
                sequence_rows=rows,
                force_chunks=chunks,
                chunk_rows=rows,
            ),
            tpu.VMEM(
                (c.q_lora, dp * HEADS_PER_RANK * (c.nope + TILE)), jnp.bfloat16
            ),  # q_buf
            tpu.VMEM((heads, c.nope, c.kv_lora), jnp.bfloat16),  # w_uk_buf
            tpu.VMEM((heads, c.kv_lora, c.v_head), jnp.bfloat16),  # w_uv_buf
            tpu.VMEM(
                (2, heads * c.v_head, CHUNK)
                if streamed or dp > 1
                else (heads * c.v_head, c.hidden),
                jnp.bfloat16,
            ),  # o_buf (streamed: two chunk slots)
            tpu.VMEM((c.hidden, MLP_PADDED), jnp.bfloat16),  # gate_buf
            tpu.VMEM((c.hidden, MLP_PADDED), jnp.bfloat16),  # up_buf
            tpu.VMEM(
                (2, MLP_PADDED, CHUNK) if streamed else (MLP_PADDED, c.hidden),
                jnp.bfloat16,
            ),  # down_buf
            tpu.VMEM((c.markov_rank, c.vocab // RANKS), jnp.bfloat16),  # w2t_buf
            tpu.VMEM((2, r, 16, c.markov_rank), jnp.bfloat16),  # w1_blk
            tpu.VMEM((16, c.hidden), jnp.bfloat16),  # embed_blk
            tpu.VMEM((r, ROW_PACKED, 128), jnp.float32),  # bcast_buf
            tpu.VMEM((2, r, 16, 2 * c.rope), jnp.float32),  # rope_blk (block, context)
            tpu.VMEM((t, c.vocab // RANKS), jnp.float32),  # logits_vmem
            tpu.VMEM((t, CHUNK), jnp.float32),  # fc_part
            tpu.VMEM((r, 128), jnp.int32),  # tokens_v
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # stage
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # local_recv
            tpu.VMEM((2, 4, packed, 128), jnp.float32),  # host_packed
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # gather_recv
            tpu.VMEM((2, RANKS, r, 128), jnp.int32),  # tiny_recv
            tpu.VMEM((RANKS, 8, 128), jnp.int32),  # argmax_recv: rows along lanes
            tpu.VMEM(
                (r // dp, 2 * rows, LANES), jnp.float32
            ),  # keys_vmem (the group's requests)
            tpu.VMEM((r // dp, 1, 2 * rows), jnp.int32),  # valid_vmem
            tpu.VMEM((c.layers, t, LANES), jnp.bfloat16),  # latents_vmem
            tpu.VMEM(
                (2, c.hidden, TILE), jnp.bfloat16
            ),  # tile_buf: q_a | kv_a projection tiles
            tpu.VMEM(
                (2, RANKS, t * TILE // 256, 128), jnp.float32
            ),  # tile_recv (packed BF16)
            # The two-stage DCP combine (chunked calls; see mla_dcp.host_combine).
            tpu.VMEM(
                (ring, hosts, HEADS_PER_RANK * rows, mla_dcp.LANES), jnp.bfloat16
            ),  # stage2_recv
            tpu.VMEM(
                (hosts, HEADS_PER_RANK * rows, mla_dcp.LANES), jnp.bfloat16
            ),  # stage2_send
            tpu.SemaphoreType.DMA((2, ring)),  # stage2_sems
            tpu.SemaphoreType.DMA((19 + 5 * r,)),
            tpu.SemaphoreType.DMA((len(SMALL_NAMES),)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
        )

        body = _make_body(c, block, r, rows, pages, owner_table, plan, dp)
        operands = (aux, logits, previous, *[weights[n] for n in WEIGHT_NAMES], *pools)
        in_specs = [pl.BlockSpec(memory_space=tpu.HBM) for _ in operands]
        return pl.pallas_call(
            body,
            name="dspark_mla_fused_draft",
            out_shape=(
                jax.ShapeDtypeStruct((r, 128), jnp.int32),
                jax.ShapeDtypeStruct((c.layers, t, LANES), jnp.bfloat16),
            ),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=in_specs,
                out_specs=[
                    pl.BlockSpec(memory_space=tpu.HBM),
                    pl.BlockSpec(memory_space=tpu.HBM),
                ],
                scratch_shapes=scratch,
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=26, vmem_limit_bytes=64 * 1024**2
            ),
        )(scalars, *operands)

    return call
