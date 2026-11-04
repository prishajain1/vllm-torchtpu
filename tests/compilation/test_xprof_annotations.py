# Copyright 2026 Google LLC
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

from collections import OrderedDict

import torch
import torch.fx as fx

from vllm_torchtpu.compilation.xprof_annotations import (
    NodeAnnotation,
    build_propagated_node_annotations,
    clean_op_name,
    compilation_context,
    derive_xla_module_name,
    extract_fx_node_scope,
    parse_fx_stack_frames,
    rewrite_mlir_debug_locations,
    sanitize_mlir_symbol,
)


class TestXprofAnnotations:
    def test_extract_fx_node_scope_normalizes_dynamo_fqns_and_skips_wrappers(self):
        nn_stack = OrderedDict(
            [
                ("root", ("L['self']", "Gemma4TextModel")),
                ("layers_list", ("L['self'].layers", torch.nn.ModuleList)),
                (
                    "layer0",
                    (
                        "getattr(L['self'].layers, '0')",
                        "vllm.model_executor.models.gemma4.Gemma4DecoderLayer",
                    ),
                ),
                (
                    "mlp",
                    (
                        "getattr(L['self'].layers, '0').mlp",
                        "vllm.model_executor.models.gemma4.Gemma4MLP",
                    ),
                ),
                (
                    "gate_up",
                    (
                        "getattr(L['self'].layers, '0').mlp.gate_up_proj",
                        "MergedColumnParallelLinear",
                    ),
                ),
            ]
        )
        scope = extract_fx_node_scope(nn_stack)
        assert scope == (
            "layers.0[Gemma4DecoderLayer]/"
            "mlp[Gemma4MLP]/"
            "gate_up_proj[MergedColumnParallelLinear]"
        )

    def test_extract_fx_node_scope_sanitizes_xla_special_chars(self):
        nn_stack = OrderedDict(
            [
                ("root", ("L['self']", "Model@v1:2")),
                ("sub", ("L['self'].attn:0(q)", "Attn(Core)")),
            ]
        )
        scope = extract_fx_node_scope(nn_stack)
        assert "@" not in scope
        assert ":" not in scope
        assert "(" not in scope
        assert ")" not in scope
        assert scope == "attn_0[q][Attn[Core]]"

    def test_extract_fx_node_scope_falls_back_to_layer_name_arg(self):
        scope = extract_fx_node_scope(
            None, fallback_args=("model.layers.12.self_attn.attn",)
        )
        assert scope == "model/layers.12/self_attn/attn"

    def test_parse_fx_stack_frames_strips_compiler_and_module_plumbing(self):
        trace = """
  File "/opt/vllm/vllm/compilation/backends.py", line 759, in __call__
    return self.compiled_graph(*args)
  File "/opt/vllm/vllm/model_executor/models/gemma4.py", line 1493, in forward
    hidden_states, residual = layer(positions, hidden_states, residual)
  File "/opt/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/vllm/vllm/model_executor/models/gemma4.py", line 1120, in forward
    hidden_states = self.mlp(hidden_states)
  File "/opt/vllm/vllm/model_executor/models/gemma4.py", line 345, in forward
    gate_up, _ = self.gate_up_proj(x)
  File "/opt/vllm_torchtpu/layers/adapter/quantization/unquantized.py", line 139, in apply
    return F.linear(x, layer.weight, bias)
"""
        frames = parse_fx_stack_frames(trace)
        assert frames == [
            ("/opt/vllm/vllm/model_executor/models/gemma4.py", 1493, "forward"),
            ("/opt/vllm/vllm/model_executor/models/gemma4.py", 1120, "forward"),
            ("/opt/vllm/vllm/model_executor/models/gemma4.py", 345, "forward"),
            (
                "/opt/vllm_torchtpu/layers/adapter/quantization/unquantized.py",
                139,
                "apply",
            ),
        ]

    def test_clean_op_name_dedups_and_strips_jit_wrappers(self):
        assert clean_op_name("mm/mm:") == "mm"
        assert clean_op_name("to_copy/copy_/as_strided:") == "to_copy/copy_/as_strided"
        assert (
            clean_op_name("dot_general(jit(run_model)/mn,np->mp)")
            == "mn,np->mp/dot_general"
        )

    def test_propagated_node_annotations_fill_unannotated_helper_nodes(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        cast_x = graph.call_function(torch.ops.aten._to_copy.default, (x,))
        mm = graph.call_function(torch.ops.aten.mm.default, (cast_x, w))
        mm.meta["nn_module_stack"] = OrderedDict(
            [
                ("root", ("L['self']", "Gemma4TextModel")),
                ("l0", ("L['self'].layers[0]", "Gemma4DecoderLayer")),
                ("qkv", ("L['self'].layers[0].self_attn.qkv_proj", "QKVParallelLinear")),
            ]
        )
        mm.stack_trace = (
            '  File "/vllm/models/gemma4.py", line 280, in forward\n'
            "    qkv, _ = self.qkv_proj(x)\n"
        )
        view_out = graph.call_function(torch.ops.aten.view.default, (mm, [16, 256]))
        graph.output((view_out,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        anns = build_propagated_node_annotations(gm)
        expected_scope = (
            "layers.0[Gemma4DecoderLayer]/self_attn.qkv_proj[QKVParallelLinear]"
        )
        assert anns[cast_x].scope_path == expected_scope
        assert anns[mm].scope_path == expected_scope
        assert anns[view_out].scope_path == expected_scope
        assert anns[view_out].frames == (("/vllm/models/gemma4.py", 280, "forward"),)

    def test_derive_xla_module_name_includes_model_bucket_and_layer_span(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        n0 = graph.call_function(torch.relu, (x,))
        n0.meta["nn_module_stack"] = OrderedDict(
            [
                ("root", ("L['self']", "Gemma4TextModel")),
                ("l0", ("L['self'].layers.0.self_attn", "Gemma4Attention")),
            ]
        )
        n1 = graph.call_function(torch.relu, (n0,))
        n1.meta["nn_module_stack"] = OrderedDict(
            [
                ("root", ("L['self']", "Gemma4TextModel")),
                ("l59", ("L['self'].layers.59.mlp", "Gemma4MLP")),
            ]
        )
        graph.output((n1,))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with compilation_context(token_bucket=16384):
            mod_name = derive_xla_module_name(gm, submod_name="submod_0")

        assert mod_name == "Gemma4TextModel_T16384_L00_to_L59_submod_0"
        assert sanitize_mlir_symbol("123-bad@name:here") == "m_123_bad_name_here"

        # Dynamo omits root L['self'] from nn_module_stack; ensure embed_tokens
        # does not hijack the model name when Gemma4DecoderLayer is present.
        dynamo_graph = fx.Graph()
        dx = dynamo_graph.placeholder("x")
        emb = dynamo_graph.call_function(torch.relu, (dx,))
        emb.meta["nn_module_stack"] = OrderedDict(
            [("emb", ("L['self'].embed_tokens", "VocabParallelEmbedding"))]
        )
        dl0 = dynamo_graph.call_function(torch.relu, (emb,))
        dl0.meta["nn_module_stack"] = OrderedDict(
            [
                ("l0", ("L['self'].layers.0", "Gemma4DecoderLayer")),
                ("mlp", ("L['self'].layers.0.mlp", "Gemma4MLP")),
            ]
        )
        dl59 = dynamo_graph.call_function(torch.relu, (dl0,))
        dl59.meta["nn_module_stack"] = OrderedDict(
            [("l59", ("L['self'].layers.59", "Gemma4DecoderLayer"))]
        )
        dynamo_graph.output((dl59,))
        dynamo_gm = fx.GraphModule(torch.nn.Module(), dynamo_graph)
        with compilation_context(token_bucket=256):
            assert (
                derive_xla_module_name(dynamo_gm, submod_name="submod_0")
                == "Gemma4Model_T256_L00_to_L59_submod_0"
            )

    def test_rewrite_mlir_debug_locations_rewrites_aten_and_pallas_ops(self):
        raw_mlir = """#loc = loc(unknown)
module @"tt_jit_<eval_with_key>.0 from gemma4.py:10 in forward" attributes {mhlo.num_partitions = 2 : i32} {
  func.func @main(%arg0: tensor<64x128xbf16> loc(unknown), %arg1: tensor<128x256xbf16> loc(unknown)) -> tensor<64x256xbf16> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16> loc(#loc10)
    %1 = stablehlo.custom_call @tpu_custom_call(%0) : (tensor<64x256xbf16>) -> tensor<64x256xbf16> loc(#loc14)
    return %1 : tensor<64x256xbf16> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/site-packages/torch_tpu/_internal/export/export.py":534:0)
#loc2 = loc("/site-packages/torch_tpu/_internal/export/export.py":207:0)
#loc3 = loc("__vllm_xprof_fx_0__":2:0)
#loc4 = loc("__vllm_xprof_fx_1__":2:0)
#loc5 = loc("/site-packages/torch/_library/custom_ops.py":502:0)
#loc6 = loc("/vllm_torchtpu/kernels/experimental/batched_rpa/wrapper.py":161:0)
#loc7 = loc("fx_to_mlir"(#loc1))
#loc8 = loc("run_node"(#loc2))
#loc9 = loc(callsite(#loc8 at #loc7))
#loc_fx0 = loc("fx_0"(#loc3))
#loc_cs0 = loc(callsite(#loc_fx0 at #loc9))
#loc10 = loc("mm/mm"(#loc_cs0))
#loc_fx1 = loc("fx_1"(#loc4))
#loc_cs1 = loc(callsite(#loc_fx1 at #loc9))
#loc11 = loc("call_custom"(#loc5))
#loc12 = loc(callsite(#loc11 at #loc_cs1))
#loc13 = loc("rpa_wrapper"(#loc6))
#loc14 = loc(callsite(#loc13 at #loc12))
"""
        marker_table = {
            0: NodeAnnotation(
                scope_path="layers.0[Gemma4DecoderLayer]/mlp[Gemma4MLP]/gate_up_proj[MergedColumnParallelLinear]",
                op_name="mm",
                frames=(
                    ("/vllm/models/gemma4.py", 1120, "forward"),
                    ("/vllm/models/gemma4.py", 345, "forward"),
                    ("/vllm_torchtpu/layers/unquantized.py", 139, "apply"),
                ),
            ),
            1: NodeAnnotation(
                scope_path="layers.0[Gemma4DecoderLayer]/self_attn[Gemma4Attention]/attn[Attention]",
                op_name="rpa_kernel_batched_0",
                frames=(
                    ("/vllm/models/gemma4.py", 1120, "forward"),
                    ("/vllm_torchtpu/layers/attention.py", 1137, "rpa_kernel_impl"),
                ),
            ),
        }

        rewritten = rewrite_mlir_debug_locations(
            raw_mlir,
            marker_table,
            module_name="Gemma4TextModel_T16384_L00_submod_0",
            donated_inputs=[0],
        )

        assert (
            "module @Gemma4TextModel_T16384_L00_submod_0 "
            "attributes {mhlo.num_partitions = 2 : i32} {" in rewritten
        )
        assert (
            "%arg0: tensor<64x128xbf16> {jax.buffer_donor = true} loc(unknown)"
            in rewritten
        )
        assert "__vllm_xprof_fx_" not in rewritten
        assert "torch_tpu/_internal/export/export.py" not in rewritten
        assert "torch/_library/custom_ops.py" not in rewritten
        assert (
            '"layers.0[Gemma4DecoderLayer]/mlp[Gemma4MLP]/'
            'gate_up_proj[MergedColumnParallelLinear]/mm"('
            in rewritten
        )
        assert (
            '"layers.0[Gemma4DecoderLayer]/self_attn[Gemma4Attention]/'
            'attn[Attention]/rpa_kernel_batched_0"('
            in rewritten
        )
        assert '"/vllm/models/gemma4.py":1120:0' in rewritten
        assert '"/vllm_torchtpu/kernels/experimental/batched_rpa/wrapper.py":161:0' in rewritten
