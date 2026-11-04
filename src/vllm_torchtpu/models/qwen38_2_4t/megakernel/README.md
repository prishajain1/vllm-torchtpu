# Qwen3.8-2.4T TP32 x DCP8 NVFP4 Megakernel (`qwen38_2_4t`)

Self-contained megakernel model folder for Qwen3.8-2.4T (`num_hidden_layers == 92`,
`hidden_size == 8192`, `3:1` GDN + Full Attention, `512` routed experts top-`8` +
`1` shared expert) under `TP=32 x DCP=8` and `NVFP4` quantization.

## Runtime Contract (`SPEC`)

- **Package**: `vllm_torchtpu.models.qwen38_2_4t.megakernel`
- **Shorthand**: `"qwen38_2_4t"` (auto-selected under `VLLM_TPU_MEGAKERNEL=1` when
  `num_hidden_layers == 92` and `hidden_size == 8192`)
- **Collective domain**: `DOMAIN_TP_X_DCP` (`tp_x_dcp`)
- **Hooks**:
  - `create = ".adapter:create"`
  - `layout_request = ".adapter:layout_request"` (`block_size=128`, `mamba_slot_tokens=640`)
  - `load_tensor = ".weights:load_tensor"` (captures pre-pack expert weights and scales)

## Composed Substrate Primitives

- **Hierarchical TP32 All-Reduce (`dma.Qwen38HierAllReduce`)**: Composes
  `PeerTopology(rank=me, world_size=32, group_size=4)` and
  `make_remote_copy_batch` from `vllm_torchtpu.megakernel.collectives` for a 2-stage
  `4x8` reduction (4-rank intra-tray ring followed by 8-way XOR reduction).
- **DCP=8 Query & LSE Exchanges**: Uses `make_remote_copy_batch` across the
  8-rank DCP group (`(me // 8) * 8 + peer`).
- **Stage Scratchpad Reuse (`plan.build_stage_arena`)**: Uses `VmemArena` and
  `arena.view(pool_ref, slice_handle)` from `vllm_torchtpu.megakernel.dma` to pack
  stage-disjoint down-projection buffers (`wo_buf`, `wout_buf`, `wse2_buf`) onto a
  shared scratchpad interval.
