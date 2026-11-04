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

import importlib
import os
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed._functional_collectives
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
from vllm.distributed import parallel_state

from vllm_torchtpu.compilation import tpu_compiler as tpu_compiler_mod
from vllm_torchtpu.compilation.tpu_compiler import (
    _canonicalize_process_group_name,
    _graph_fingerprint,
    _normalized_module_code,
    wait_for_peer_artifact,
)
from vllm_torchtpu.env_override import _TPU_VMEM_BYTES_PER_TENSOR_CORE


class TestTpuCompilerCache:
    @staticmethod
    def _config(*, max_num_seqs=16, num_gpu_blocks=1024):
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                gpu_memory_utilization=0.9,
                num_gpu_blocks=num_gpu_blocks,
                num_gpu_blocks_override=None,
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2048,
                max_num_seqs=max_num_seqs,
            ),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            speculative_config=None,
        )

    def test_tpu_hash_covers_derived_shapes(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            base = compute_tpu_compilation_hash(self._config())
            different_batch = compute_tpu_compilation_hash(
                self._config(max_num_seqs=32)
            )
            different_kv = compute_tpu_compilation_hash(
                self._config(num_gpu_blocks=2048)
            )

        assert base != different_batch
        assert base != different_kv

    def test_tpu_hash_covers_plugin_compile_env(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"ONEHOT_MOE_PERMUTE_THRESHOLD": "0"}):
                disabled = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"ONEHOT_MOE_PERMUTE_THRESHOLD": "4096"}):
                enabled = compute_tpu_compilation_hash(self._config())

        assert disabled != enabled

    def test_tpu_hash_ignores_the_phased_profiler_switch(self):
        """Selecting a profiler must not bust the compile cache: otherwise a
        profiled run recompiles everything, and so does the next plain one."""
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"USE_PHASED_PROFILER": "false"}):
                unprofiled = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"USE_PHASED_PROFILER": "true"}):
                phased = compute_tpu_compilation_hash(self._config())

        assert unprofiled == phased

    def test_tpu_hash_covers_native_compile_env(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/a"}):
                first = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/b"}):
                second = compute_tpu_compilation_hash(self._config())

        assert first != second

    def test_tpu_hash_covers_compiler_versions(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        versions = {
            "torch-tpu": "1",
            "jax": "1",
            "jaxlib": "1",
            "libtpu": "1",
        }
        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch(
                "vllm_torchtpu.compilation.tpu_compiler.importlib_metadata.version",
                side_effect=versions.get,
            ):
                base = compute_tpu_compilation_hash(self._config())

            for package in versions:
                changed_versions = {**versions, package: "2"}
                with patch(
                    "vllm_torchtpu.compilation.tpu_compiler.importlib_metadata.version",
                    side_effect=changed_versions.get,
                ):
                    changed = compute_tpu_compilation_hash(self._config())
                assert base != changed

    def test_tpu_hash_covers_effective_data_parallel_size(self):
        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        with patch(
            "vllm_torchtpu.compilation.tpu_compiler._iter_runtime_cache_key_files",
            return_value=[],
        ):
            with patch.dict(os.environ, {"TORCH_TPU_DP_SIZE": "1"}):
                dp1 = compute_tpu_compilation_hash(self._config())
            with patch.dict(os.environ, {"TORCH_TPU_DP_SIZE": "4"}):
                dp4 = compute_tpu_compilation_hash(self._config())

        assert dp1 != dp4

    def test_runtime_hash_covers_custom_op_boundaries(self):
        from vllm_torchtpu.compilation import tpu_compiler

        source_root = Path(tpu_compiler.__file__).resolve().parents[2]
        hashed = {
            path.relative_to(source_root).as_posix()
            for path in tpu_compiler._iter_runtime_cache_key_files(source_root)
        }
        custom_op_boundaries = {
            path.relative_to(source_root).as_posix()
            for path in (source_root / "vllm_torchtpu").rglob("*.py")
            if "pallas.jax_op" in path.read_text()
        }
        assert custom_op_boundaries <= hashed
        assert {
            "vllm_torchtpu/kernels/quantized_matmul/blockwise_kernel.py",
            "vllm_torchtpu/kernels/pool_adapters.py",
            "vllm_torchtpu/models/vllm/kimi_k3/collective_ops.py",
            "vllm_torchtpu/models/vllm/kimi_k3/layers.py",
        } <= hashed

    def test_serving_only_kv_knobs_stay_out_of_the_compile_cache_key(self):
        """Serving-time knobs must stay out of the compile-cache key.

        Every `envs.py` entry not in `_TPU_COMPILE_ENV_IGNORED` is hashed, so
        a knob added there recompiles the world once on upgrade and again
        every time an operator retunes it.
        """
        from vllm_torchtpu.compilation.tpu_compiler import (
            _TPU_COMPILE_ENV_IGNORED,
            _tpu_compile_env_factors,
        )

        serving_only = {
            "TPU_IPC_SOCKET_DIR",
            "TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS",
            "TPU_KV_COORD_EXECUTOR_MAX_WORKERS",
            "TPU_KV_LATENCY_LOG_INTERVAL",
            "TPU_KV_PIN_SHM",
            "TPU_KV_SHM_POOL_GB",
            "TPU_KV_STAGE_WAITER_POOL_SIZE",
            "TPU_KV_STAGE_WAIT_TIMEOUT_SECS",
            "TPU_KV_TRANSFER_CHANNEL_NUMBER",
            "TPU_KV_TRANSFER_NAMESPACE",
            "TPU_KV_TRANSFER_PORT",
            "TPU_KV_WARMUP_ENABLED",
            "TPU_NODE_ID",
            "TPU_P2P_WAIT_PULL_TIMEOUT",
            "TPU_RAIDEN_INLINE_LOAD",
            "TPU_RAIDEN_POOL_STAGING_LEASES",
            "TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT",
            "TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S",
            "TPU_RAIDEN_STAGE3_STATUS_PROBE_S",
            "TPU_RAIDEN_TEST_REGISTRATION_DELAY_S",
            "TPU_RAIDEN_TRANSFER_NUM_SLOTS",
            "TPU_SIDE_CHANNEL_PORT",
            "TPU_USE_RAIDEN_CONNECTOR",
            "DEBUG_TPU_LOCAL_RANK_OFFSET",
            "RAIDEN_DISABLE_SINGLETON_WORKER",
            "RAIDEN_SHM_KEY",
            "SPEC_WARMUP",
            "TORCH_TPU_BASE_PORT",
            "TORCH_TPU_MP_RENDEZVOUS_PORT",
            "TPU_LOCAL_RANK_OFFSET",
            "TPU_SHARDED_LOAD_SYNC_EVERY",
            "VLLM_TORCHTPU_IPC_KEY",
            "VLLM_TPU_DEBUG_PCP_LAYOUT",
            "VLLM_TPU_OFFLOAD_SAVE_RETRIES",
            "VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S",
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        }

        assert serving_only <= _TPU_COMPILE_ENV_IGNORED
        assert not serving_only & set(_tpu_compile_env_factors())

    def test_every_registry_entry_is_hashed_or_deliberately_ignored(self):
        """The two sets must partition the registry, so a new entry forces a
        deliberate hash-or-ignore choice, and a stale exemption cannot sit
        waiting for a later knob to reuse its name."""
        from vllm_torchtpu import envs
        from vllm_torchtpu.compilation.tpu_compiler import (
            _TPU_COMPILE_ENV_IGNORED,
            _tpu_compile_env_factors,
        )

        registry = set(envs.environment_variables)
        stale = _TPU_COMPILE_ENV_IGNORED - registry
        assert not stale, f"ignored but no longer in envs.py: {sorted(stale)}"
        assert set(_tpu_compile_env_factors()) | _TPU_COMPILE_ENV_IGNORED == registry

    def test_aot_hash_includes_tpu_compiler_hash(self):
        from vllm.compilation import caching

        from vllm_torchtpu import _patch_vllm_aot_compile_cache_key

        def upstream_factors(_):
            return ["upstream"]

        with patch.object(caching, "aot_compile_hash_factors", upstream_factors):
            _patch_vllm_aot_compile_cache_key()
            with patch(
                "vllm_torchtpu.compilation.tpu_compiler.compute_tpu_compilation_hash",
                return_value="tpu",
            ):
                factors = caching.aot_compile_hash_factors(self._config())

        assert factors == ["upstream", "tpu"]

    def test_env_override_mapping(self):
        """Verify that setting TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT sets
        TORCH_TPU_TIER2_COMPILATION_CACHE default.
        """
        env_mock = {
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/tmp/test_tier3_cache_env",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            # Reload env_override to re-execute its top-level statements
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            assert (
                os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE") == "tpu_tier2_cache"
            )

    @staticmethod
    def _sc_offload_flags() -> list[str]:
        return [
            token
            for token in os.environ.get("LIBTPU_INIT_ARGS", "").split()
            if "offload_min_size_in_bytes" in token
        ]

    @pytest.mark.parametrize(
        "value,expected_threshold",
        [
            ("12345", 12345),
            ("auto", _TPU_VMEM_BYTES_PER_TENSOR_CORE["v6e"]),
            ("", _TPU_VMEM_BYTES_PER_TENSOR_CORE["v6e"]),
            ("0", None),
        ],
    )
    def test_env_override_sc_offload_threshold(self, value, expected_threshold):
        """An explicit value is used as given, unset/empty/auto resolves
        from the chip family, and 0 sets no flag at all."""
        env_mock = {
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": value,
            "TPU_ACCELERATOR_TYPE": "v6e-8",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            expected = (
                []
                if expected_threshold is None
                else [
                    f"--xla_tpu_sparse_core_{op}_offload_min_size_in_bytes="
                    f"{expected_threshold}"
                    for op in ("all_reduce", "all_gather")
                ]
            )
            assert self._sc_offload_flags() == expected

    @pytest.mark.parametrize("value", ["banana", "-1"])
    def test_env_override_sc_offload_bad_value_raises(self, value):
        """A value that is not "auto" or a non-negative integer stops the
        package import with an error that names the variable, rather than
        silently running without the SC-offload flag."""
        env_mock = {
            "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": value,
            "TPU_ACCELERATOR_TYPE": "v6e-8",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            with pytest.raises(
                ValueError, match="SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES"
            ):
                importlib.reload(vllm_torchtpu.env_override)

    def test_envs_module_imports_only_the_standard_library(self):
        """env_override.py imports envs while the package's own __init__ is
        still running, which is only safe while envs.py depends on nothing
        in this package."""
        import ast
        import sys

        import vllm_torchtpu.envs

        tree = ast.parse(Path(vllm_torchtpu.envs.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= sys.stdlib_module_names, (
            f"envs.py imports non-stdlib modules: "
            f"{sorted(imported - sys.stdlib_module_names)}"
        )

    def test_env_override_disables_breakable_cudagraph(self):
        """Verify that the TPU plugin always disables breakable cudagraph."""
        with patch.dict(os.environ, {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"}):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"

    def test_env_override_no_clobber(self):
        """Verify that existing native variables are not overwritten by env_override."""
        env_mock = {
            "VLLM_XLA_CACHE_PATH": "/tmp/test_xla_cache_env",
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "/custom/tier3/path",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/custom/tier3/path",
            "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "custom_tier2",
            "TORCH_TPU_TIER2_COMPILATION_CACHE": "custom_tier2",
        }
        with patch.dict(os.environ, env_mock, clear=True):
            import vllm_torchtpu.env_override

            importlib.reload(vllm_torchtpu.env_override)

            assert (
                os.environ.get("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT")
                == "/custom/tier3/path"
            )
            assert (
                os.environ.get("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")
                == "/custom/tier3/path"
            )

            assert (
                os.environ.get("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE")
                == "custom_tier2"
            )
            assert os.environ.get("TORCH_TPU_TIER2_COMPILATION_CACHE") == "custom_tier2"

    def test_tpu_compilation_handle_serialization(self):
        """Round-trip the artifact payload TpuCompilerAdaptor.compile writes."""
        import pickle

        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilationHandle

        handle = TpuCompilationHandle(
            key="test_key_123",
            entry="mock_entry",
            signature=(("torch.int32", (16,)),),
            was_wrapped=True,
        )
        loaded = pickle.loads(pickle.dumps(handle))
        assert isinstance(loaded, TpuCompilationHandle)
        assert loaded.key == "test_key_123"
        assert loaded.entry == "mock_entry"
        assert loaded.signature == (("torch.int32", (16,)),)
        assert loaded.was_wrapped is True

    def test_bundled_replay_does_not_copy_the_entry(self):
        """Replay must not load a second executable for reserialization."""
        from vllm_torchtpu.compilation.tpu_compiler import _deserialize_entry

        class Entry:
            guards_expr = None
            sanitized_aot_config = None

            def __deepcopy__(self, memo):
                raise AssertionError("replay must not deepcopy the entry")

            def wrap_post_compile(self, example_inputs, aot_config, options):
                return lambda boxed_args: sum(boxed_args)

        # Use Torch's real deserializer and wrapper, without initializing TPU.
        runtime = _deserialize_entry(Entry())
        assert runtime(2, 3, 7) == 12

    @pytest.mark.cpu_test
    @pytest.mark.parametrize("reuse_caller_mode", [False, True])
    def test_replay_retraces_factory_ops_without_allocating_real_tensors(
        self, monkeypatch, tmp_path, reuse_caller_mode
    ):
        import pickle
        from contextlib import nullcontext

        import torch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        from vllm_torchtpu.compilation import shape_variants, tpu_compiler

        graph = torch.fx.Graph()
        value = graph.call_function(
            torch.empty, args=((2, 2),), kwargs={"device": "cpu"}
        )
        graph.output(value)
        module = torch.fx.GraphModule(torch.nn.Module(), graph)
        handle = tpu_compiler.TpuCompilationHandle(
            key="factory",
            entry=module,
            signature=shape_variants.graph_signature(module),
        )
        path = tmp_path / "artifact.pkl"
        path.write_bytes(pickle.dumps(handle))
        monkeypatch.setattr(tpu_compiler, "_deserialize_entry", lambda entry: entry)
        compiler = tpu_compiler.TpuCompilerAdaptor()
        caller_mode = FakeTensorMode() if reuse_caller_mode else None

        with caller_mode if caller_mode is not None else nullcontext():
            runtime = compiler._replay(str(path), module, was_wrapped=False)
            assert runtime is not None
            result = runtime()
            assert isinstance(result, FakeTensor)
            if caller_mode is not None:
                assert result.fake_mode is caller_mode

    def test_replay_refuses_artifacts_it_cannot_vouch_for(self, tmp_path):
        """Stale or damaged artifacts recompile instead of being replayed.

        The older format stored a bare executable picked off a shared list by
        index, with nothing tying it to the graph it was saved under, so
        replaying one can run the wrong program. Restoring such a pickle leaves
        `entry` at its class default of None, and it is refused on that basis.
        """
        import pickle

        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilationHandle,
            TpuCompilerAdaptor,
        )

        compiler = TpuCompilerAdaptor()
        path = tmp_path / "artifact"

        legacy = object.__new__(TpuCompilationHandle)
        legacy.__dict__.update(key="k", executable="mock_exe")
        assert legacy.entry is None
        path.write_bytes(pickle.dumps(legacy))
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        path.write_bytes(pickle.dumps({"key": "k", "executable": "mock_exe"}))
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        path.write_bytes(b"not a pickle")
        assert compiler._replay(str(path), graph=None, was_wrapped=False) is None

        assert (
            compiler._replay(str(tmp_path / "missing"), graph=None, was_wrapped=False)
            is None
        )

    def test_compiler_initialize_cache(self):
        """Verify that TpuCompilerAdaptor sets cache_dir accurately."""
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        compiler = TpuCompilerAdaptor()
        compiler.initialize_cache(cache_dir="/tmp/test_cache_dir")
        assert compiler.cache_dir == "/tmp/test_cache_dir"

    def test_run_backend_restarts_scheduling_group_ids(self, monkeypatch):
        """A bucket's HLO, and so its cache key, must not depend on what the
        rank compiled before it."""
        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        firsts = []

        def fake_backend(graph, example_inputs):
            firsts.append(config.reserve_scheduling_group_ids(3))
            return lambda *args: args

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
        compiler = TpuCompilerAdaptor()
        compiler._run_backend(None, [])
        compiler._run_backend(None, [])
        assert firsts == [1, 1]

    @staticmethod
    def _peer_compilers(tmp_path, monkeypatch, n=2, peer_replay="1"):
        """Compilers for local ranks 0..n-1 of one host, with a fake backend
        that records which ranks really compiled. peer_replay=None leaves
        TPU_PRECOMPILE_PEER_REPLAY unset."""
        import torch
        from vllm.config.utils import Range

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

        if peer_replay is None:
            monkeypatch.delenv("TPU_PRECOMPILE_PEER_REPLAY", raising=False)
        else:
            monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", peer_replay)
        compiled, replayed = [], []

        class Compiled:
            def __init__(self, rank):
                self.rank = rank

            def __call__(self, *args):
                return (self.rank,)

            def serialize(self):
                return f"entry-from-rank-{self.rank}"

        def compilers():
            for rank in range(n):
                compiler = TpuCompilerAdaptor()
                compiler.initialize_cache(
                    cache_dir=str(tmp_path / "hash" / f"rank_{rank}_0" / "backbone")
                )
                compiler.rank = rank
                yield compiler

        current = {}

        def fake_backend(graph, example_inputs):
            compiled.append(current["rank"])
            return Compiled(current["rank"])

        def fake_deserialize(entry):
            replayed.append(entry)
            return lambda *args: (entry,)

        monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
        monkeypatch.setattr(tpu_compiler, "_deserialize_entry", fake_deserialize)
        monkeypatch.setattr(tpu_compiler, "_PEER_POLL_S", 0.01)

        def compile(compiler, graph=None):
            current["rank"] = compiler.rank
            if graph is None:
                graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)
            return compiler.compile(
                graph, [], {}, Range(start=1, end=4096), key="artifact_bucket"
            )

        return list(compilers()), compile, compiled, replayed

    def test_a_peer_artifact_is_replayed_instead_of_compiled(
        self, tmp_path, monkeypatch
    ):
        """The second rank of a host replays what the first one saved, and its
        own cache entry is the same file, not a copy."""
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a)
        fn, handle = compile(b)

        assert compiled == [0]
        assert replayed == ["entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        key, path, _ = handle
        assert path == os.path.join(b.cache_dir, key)
        assert os.path.samefile(path, os.path.join(a.cache_dir, key))
        assert sorted(os.listdir(tmp_path / "hash")) == ["rank_0_0", "rank_1_0"]
        assert os.listdir(b.cache_dir) == [key]

    def test_a_peer_artifact_of_another_graph_is_not_replayed(
        self, tmp_path, monkeypatch
    ):
        """vLLM's key names a bucket, not a graph: a rank that traced
        something else, e.g. with its rank baked in, compiles its own."""
        import torch

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a, torch.fx.symbolic_trace(lambda x: x * 2 + 1))
        compile(b, torch.fx.symbolic_trace(lambda x: x * 2 + 2))

        assert compiled == [0, 1]
        assert replayed == []
        assert not os.path.samefile(
            os.path.join(a.cache_dir, "artifact_bucket"),
            os.path.join(b.cache_dir, "artifact_bucket"),
        )

    def test_a_rank_waits_for_the_peer_that_compiles_a_bucket(
        self, tmp_path, monkeypatch
    ):
        import threading

        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        result = {}

        def waiter():
            with wait_for_peer_artifact(30.0):
                result["fn"], _ = compile(b)

        thread = threading.Thread(target=waiter)
        thread.start()
        thread.join(0.2)
        assert thread.is_alive()
        compile(a)
        thread.join(30.0)

        assert not thread.is_alive()
        assert compiled == [0]
        assert result["fn"]() == "entry-from-rank-0"

    def test_a_rank_compiles_itself_when_no_peer_saves_in_time(
        self, tmp_path, monkeypatch
    ):
        from vllm_torchtpu.compilation.tpu_compiler import wait_for_peer_artifact

        (_, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        with wait_for_peer_artifact(0.05):
            compile(b)
        assert compiled == [1]
        assert replayed == []

    def test_no_peer_replay_outside_a_rank_cache_dir(self, tmp_path, monkeypatch):
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        a.initialize_cache(cache_dir=str(tmp_path / "x" / "backbone"))
        b.initialize_cache(cache_dir=str(tmp_path / "y" / "backbone"))
        compile(a)
        compile(b)
        assert compiled == [0, 1]

    @pytest.mark.parametrize("peer_replay", ["0", None])
    def test_peer_replay_off_compiles_every_rank_itself(
        self, tmp_path, monkeypatch, peer_replay
    ):
        """TPU_PRECOMPILE_PEER_REPLAY off, explicitly or by default (None): no
        peer lookup, no wait, no link, and no fingerprint recorded; each rank
        compiles and saves its own."""
        import pickle

        from vllm_torchtpu.compilation import tpu_compiler
        from vllm_torchtpu.compilation.tpu_compiler import (
            TpuCompilerAdaptor,
            wait_for_peer_artifact,
        )

        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch, peer_replay=peer_replay
        )

        def no_peer_lookup(*args, **kwargs):
            raise AssertionError("looked for a peer artifact")

        monkeypatch.setattr(TpuCompilerAdaptor, "_peer_paths", no_peer_lookup)
        monkeypatch.setattr(TpuCompilerAdaptor, "_link_artifact", no_peer_lookup)
        monkeypatch.setattr(tpu_compiler, "_graph_fingerprint", no_peer_lookup)

        compile(a)
        with wait_for_peer_artifact(30.0):
            _, handle = compile(b)

        assert compiled == [0, 1]
        assert replayed == []
        key, path, _ = handle
        assert not os.path.samefile(path, os.path.join(a.cache_dir, key))
        with open(path, "rb") as f:
            assert pickle.load(f).fingerprint is None

    @staticmethod
    def _trace_all_reduce_graph(
        group_name: str, reduce_op: str = "sum"
    ) -> fx.GraphModule:
        def forward(x: torch.Tensor) -> torch.Tensor:
            reduced = torch.ops._c10d_functional.all_reduce(x, reduce_op, group_name)
            waited = torch.ops._c10d_functional.wait_tensor(reduced)
            return waited + 1.0

        return fx.symbolic_trace(forward)

    @staticmethod
    def _trace_multi_collective_graph(
        tp_group: str,
        ep_group: str,
        reduce_op: str = "sum",
        group_size: int = 2,
    ) -> fx.GraphModule:
        def forward(x: torch.Tensor) -> torch.Tensor:
            scattered = torch.ops._c10d_functional.reduce_scatter_tensor(
                x, reduce_op, group_size, tp_group
            )
            scattered_wait = torch.ops._c10d_functional.wait_tensor(scattered)
            routed = torch.ops._c10d_functional.all_to_all_single(
                scattered_wait, [2, 2], [2, 2], ep_group
            )
            routed_wait = torch.ops._c10d_functional.wait_tensor(routed)
            gathered = torch.ops._c10d_functional.all_gather_into_tensor(
                routed_wait, group_size, tp_group
            )
            gathered_wait = torch.ops._c10d_functional.wait_tensor(gathered)
            reduced = torch.ops._c10d_functional.all_reduce(
                gathered_wait, reduce_op, tp_group
            )
            return torch.ops._c10d_functional.wait_tensor(reduced)

        return fx.symbolic_trace(forward)

    @staticmethod
    def _install_group_stand_ins(monkeypatch):
        class _FakeDeviceGroup:
            def __init__(self, group_name: str) -> None:
                self.group_name = group_name

        class _FakeGroup:
            def __init__(
                self,
                unique_name: str,
                ranks: list[int],
                group_ranks: list[list[int]],
                device_group_name: str | None = None,
            ) -> None:
                self.unique_name = unique_name
                self.ranks = ranks
                self.world_size = len(ranks)
                self.group_ranks = group_ranks
                self.device_group = (
                    _FakeDeviceGroup(device_group_name)
                    if device_group_name is not None
                    else None
                )

        tp_2_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
        tp_strided_groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
        dp_2_groups = [[0, 2], [1, 3], [4, 6], [5, 7]]
        ep_8_groups = [[0, 1, 2, 3, 4, 5, 6, 7]]
        tp_wrap_groups = [[0, 1, 6, 7], [4, 5, 2, 3]]

        live_groups = [
            _FakeGroup(unique_name="tp:0", ranks=[0, 1], group_ranks=tp_2_groups),
            _FakeGroup(unique_name="tp:3", ranks=[6, 7], group_ranks=tp_2_groups),
            _FakeGroup(unique_name="tp:4", ranks=[0, 4], group_ranks=tp_strided_groups),
            _FakeGroup(unique_name="dp:0", ranks=[0, 2], group_ranks=dp_2_groups),
            _FakeGroup(unique_name="dp:3", ranks=[5, 7], group_ranks=dp_2_groups),
            _FakeGroup(unique_name="pcp:0", ranks=[0, 1], group_ranks=tp_2_groups),
            _FakeGroup(unique_name="pcp:3", ranks=[6, 7], group_ranks=tp_2_groups),
            _FakeGroup(unique_name="dcp:0", ranks=[0, 1], group_ranks=tp_2_groups),
            _FakeGroup(unique_name="dcp:3", ranks=[6, 7], group_ranks=tp_2_groups),
            _FakeGroup(
                unique_name="ep:0",
                ranks=[0, 1, 2, 3, 4, 5, 6, 7],
                group_ranks=ep_8_groups,
            ),
            _FakeGroup(
                unique_name="ep:3",
                ranks=[0, 1, 2, 3, 4, 5, 6, 7],
                group_ranks=ep_8_groups,
            ),
            _FakeGroup(
                unique_name="tp_wrap:0",
                ranks=[0, 1, 6, 7],
                group_ranks=tp_wrap_groups,
                device_group_name="tp_wrap_c10d_0",
            ),
            _FakeGroup(
                unique_name="tp_wrap:1",
                ranks=[4, 5, 2, 3],
                group_ranks=tp_wrap_groups,
                device_group_name="tp_wrap_c10d_1",
            ),
            _FakeGroup(
                unique_name="pcp_wrap:0",
                ranks=[0, 4],
                group_ranks=tp_strided_groups,
                device_group_name="pcp_wrap_c10d_0",
            ),
        ]
        monkeypatch.setattr(
            parallel_state,
            "_groups",
            {group.unique_name: weakref.ref(group) for group in live_groups},
        )

        pg_ranks = {
            "1": [0, 1],
            "3": [2, 3],
            "5": [4, 5],
            "7": [6, 7],
            "13": [0, 2, 4, 6],
            "15": [1, 3, 5, 7],
            "99": [0, 4],
            "ep": [0, 1, 2, 3, 4, 5, 6, 7],
        }

        def _resolve_pg(name):
            key = str(name)
            if key not in pg_ranks:
                raise KeyError(key)
            return key

        monkeypatch.setattr(c10d, "_resolve_process_group", _resolve_pg)
        monkeypatch.setattr(
            c10d, "get_process_group_ranks", lambda pg: pg_ranks[str(pg)]
        )
        return live_groups

    def test_a_peer_artifact_is_replayed_across_c10d_subgroups(
        self, tmp_path, monkeypatch
    ):
        """Ranks in distinct DP/TP subgroups trace different rank-local c10d
        group_name strings for the same collective role, yet replay the first
        rank's compiled artifact."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)
        (a, b), compile, compiled, replayed = self._peer_compilers(
            tmp_path, monkeypatch
        )
        compile(a, self._trace_all_reduce_graph("1"))
        fn, handle = compile(b, self._trace_all_reduce_graph("7"))

        assert compiled == [0]
        assert replayed == ["entry-from-rank-0"]
        assert fn() == "entry-from-rank-0"
        key, path, _ = handle
        assert a.cache_dir is not None
        assert os.path.samefile(path, os.path.join(a.cache_dir, key))

    def test_fingerprint_canonicalizes_c10d_subgroup_names_and_preserves_graph(
        self, monkeypatch
    ):
        """Multi-group and role-indexed graphs match across subgroups without
        mutating the GraphModule's code or node arguments."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)
        gm_rank0 = self._trace_multi_collective_graph(tp_group="1", ep_group="13")
        gm_rank7 = self._trace_multi_collective_graph(tp_group="7", ep_group="15")

        code_before = gm_rank0.code
        args_before = [(node.args, dict(node.kwargs)) for node in gm_rank0.graph.nodes]

        fp_rank0 = _graph_fingerprint(gm_rank0)
        fp_rank7 = _graph_fingerprint(gm_rank7)

        assert fp_rank0 is not None
        assert fp_rank0 == fp_rank7
        assert gm_rank0.code == code_before
        assert [
            (node.args, dict(node.kwargs)) for node in gm_rank0.graph.nodes
        ] == args_before

        for role in ("tp", "dp", "pcp", "dcp", "ep"):
            assert _graph_fingerprint(
                self._trace_all_reduce_graph(f"{role}:0")
            ) == _graph_fingerprint(self._trace_all_reduce_graph(f"{role}:3"))

    def test_fingerprint_with_vllm_parallel_state_and_c10d_stand_ins(self, monkeypatch):
        """Resolved vLLM GroupCoordinators and c10d ProcessGroups encode
        partition or stride geometry and differ when role, size, or layout
        differ."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)

        assert (
            _canonicalize_process_group_name("tp:0", 0)
            == "vllm_pg:0:tp:size=2:group_ranks=((0, 1), (2, 3), (4, 5), (6, 7))"
        )
        assert (
            _canonicalize_process_group_name("tp:4", 0)
            == "vllm_pg:0:tp:size=2:group_ranks=((0, 4), (1, 5), (2, 6), (3, 7))"
        )
        assert (
            _canonicalize_process_group_name("1", 0) == "c10d_pg:0:size=2:strides=(1,)"
        )
        assert _graph_fingerprint(
            self._trace_all_reduce_graph("tp:0")
        ) != _graph_fingerprint(self._trace_all_reduce_graph("tp:4"))
        assert _graph_fingerprint(
            self._trace_all_reduce_graph("tp:0")
        ) != _graph_fingerprint(self._trace_all_reduce_graph("ep:0"))

        assert _graph_fingerprint(
            self._trace_all_reduce_graph("1")
        ) != _graph_fingerprint(self._trace_all_reduce_graph("13"))
        assert _graph_fingerprint(
            self._trace_all_reduce_graph("1")
        ) != _graph_fingerprint(self._trace_all_reduce_graph("99"))

    def test_fingerprint_matches_across_wrap_around_tpu7x_subgroups(self, monkeypatch):
        """Wrap-around tpu7x subgroups (e.g. tp=(0,1,6,7) and tp=(4,5,2,3))
        share the same GroupCoordinator.group_ranks partition and match whether
        referenced by unique_name or device_group.group_name."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)

        assert _graph_fingerprint(
            self._trace_all_reduce_graph("tp_wrap:0")
        ) == _graph_fingerprint(self._trace_all_reduce_graph("tp_wrap:1"))
        assert _graph_fingerprint(
            self._trace_all_reduce_graph("tp_wrap_c10d_0")
        ) == _graph_fingerprint(self._trace_all_reduce_graph("tp_wrap_c10d_1"))
        assert _graph_fingerprint(
            self._trace_all_reduce_graph("tp_wrap_c10d_0")
        ) != _graph_fingerprint(self._trace_all_reduce_graph("pcp_wrap_c10d_0"))

    def test_fingerprint_canonicalizes_keyword_group_name_and_packet_without_default(
        self, monkeypatch
    ):
        """Keyword group_name arguments are canonicalized without mutating
        kwargs, and OpOverloadPackets lacking a 'default' overload (such as
        torch.ops.aten.add_) do not cause _node_schema to fail."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)
        assert "default" not in torch.ops.aten.add_.overloads()

        def _build_kwarg_graph(group_name: str) -> fx.GraphModule:
            graph = fx.Graph()
            x = graph.placeholder("x")
            added = graph.call_function(torch.ops.aten.add_, (x, x))
            reduced = graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                (added,),
                {"reduce_op": "sum", "group_name": group_name},
            )
            waited = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default, (reduced,)
            )
            graph.output(waited)
            return fx.GraphModule(torch.nn.Module(), graph)

        gm_rank0 = _build_kwarg_graph("1")
        gm_rank7 = _build_kwarg_graph("7")
        gm_rank_other = _build_kwarg_graph("13")
        kwargs_before = [dict(node.kwargs) for node in gm_rank0.graph.nodes]

        fp_rank0 = _graph_fingerprint(gm_rank0)
        assert fp_rank0 is not None
        assert fp_rank0 == _graph_fingerprint(gm_rank7)
        assert fp_rank0 != _graph_fingerprint(gm_rank_other)
        assert [dict(node.kwargs) for node in gm_rank0.graph.nodes] == kwargs_before

    def test_fingerprint_shares_group_ordinals_across_submodules(self, monkeypatch):
        """Group ordinals are shared across graph.named_modules() so submodules
        using distinct same-geometry groups do not reset to ordinal 0."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)

        def _build_two_submodule_graph(group_a: str, group_b: str) -> fx.GraphModule:
            root = torch.nn.Module()
            root.sub_a = self._trace_all_reduce_graph(group_a)
            root.sub_b = self._trace_all_reduce_graph(group_b)
            graph = fx.Graph()
            x = graph.placeholder("x")
            y = graph.call_module("sub_a", (x,))
            z = graph.call_module("sub_b", (y,))
            graph.output(z)
            return fx.GraphModule(root, graph)

        fp_same_0 = _graph_fingerprint(_build_two_submodule_graph("1", "1"))
        fp_same_7 = _graph_fingerprint(_build_two_submodule_graph("7", "7"))
        fp_distinct_07 = _graph_fingerprint(_build_two_submodule_graph("1", "7"))
        fp_distinct_70 = _graph_fingerprint(_build_two_submodule_graph("7", "1"))

        assert fp_same_0 is not None
        assert fp_same_0 == fp_same_7
        assert fp_distinct_07 == fp_distinct_70
        assert fp_same_0 != fp_distinct_07

    def test_non_symmetric_collective_returns_none_fingerprint(self, monkeypatch):
        """Rooted or point-to-point collectives (such as broadcast) raise
        ValueError in _normalized_module_code so _graph_fingerprint returns
        None instead of aliasing across subgroups."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)

        def _broadcast_forward(x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.broadcast(x, 0, "1")
            return torch.ops._c10d_functional.wait_tensor(out)

        gm = fx.symbolic_trace(_broadcast_forward)
        with pytest.raises(
            ValueError,
            match=(
                "Cannot canonicalize non-symmetric collective "
                "'_c10d_functional::broadcast'"
            ),
        ):
            _normalized_module_code(gm)
        assert _graph_fingerprint(gm) is None

    def test_unresolved_group_returns_none_fingerprint(self, monkeypatch):
        """An unresolvable group_name raises ValueError in
        _canonicalize_process_group_name and causes _graph_fingerprint to return
        None."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)

        with pytest.raises(
            ValueError,
            match="Cannot resolve process group 'unknown_pg'",
        ):
            _canonicalize_process_group_name("unknown_pg", 0)

        assert _graph_fingerprint(self._trace_all_reduce_graph("unknown_pg")) is None

    def test_fingerprint_negative_collision_checks(self, monkeypatch):
        """Semantically different graphs must never collide after canonicalization."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)
        fp_sum = _graph_fingerprint(self._trace_all_reduce_graph("1", reduce_op="sum"))
        fp_avg = _graph_fingerprint(self._trace_all_reduce_graph("1", reduce_op="avg"))
        assert fp_sum != fp_avg

        def _reduce_scatter_only(x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.reduce_scatter_tensor(x, "sum", 2, "1")
            return torch.ops._c10d_functional.wait_tensor(out)

        assert fp_sum != _graph_fingerprint(fx.symbolic_trace(_reduce_scatter_only))

        assert _graph_fingerprint(
            self._trace_multi_collective_graph("1", "13", group_size=2)
        ) != _graph_fingerprint(
            self._trace_multi_collective_graph("1", "13", group_size=4)
        )

        assert _graph_fingerprint(
            self._trace_multi_collective_graph(tp_group="1", ep_group="1")
        ) != _graph_fingerprint(
            self._trace_multi_collective_graph(tp_group="1", ep_group="13")
        )

        assert _graph_fingerprint(
            self._trace_all_reduce_graph("tp:0")
        ) != _graph_fingerprint(self._trace_all_reduce_graph("dp:0"))

    def test_legacy_torchtpu_pure_subgroup_vs_world_tiled_fingerprint(
        self, monkeypatch
    ):
        """Legacy TorchTPU (< 0.2.0) scopes pure subgroup graphs to each
        subgroup while canonicalizing graphs that contain a world-sized op
        across all subgroups."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)
        monkeypatch.setattr(tpu_compiler_mod, "_distributed_world_size", lambda: 8)
        monkeypatch.setattr(
            tpu_compiler_mod, "_supports_global_subgroup_tiling", lambda: False
        )

        fp_r0 = _graph_fingerprint(self._trace_all_reduce_graph("1"))
        fp_r1 = _graph_fingerprint(self._trace_all_reduce_graph("1"))
        fp_r2 = _graph_fingerprint(self._trace_all_reduce_graph("3"))
        fp_r3 = _graph_fingerprint(self._trace_all_reduce_graph("3"))
        fp_r4 = _graph_fingerprint(self._trace_all_reduce_graph("5"))
        fp_r6 = _graph_fingerprint(self._trace_all_reduce_graph("7"))

        assert fp_r0 is not None
        assert fp_r0 == fp_r1
        assert fp_r2 == fp_r3
        assert len({fp_r0, fp_r2, fp_r4, fp_r6}) == 4

        fp_world_g0 = _graph_fingerprint(
            self._trace_multi_collective_graph(tp_group="1", ep_group="ep")
        )
        fp_world_g1 = _graph_fingerprint(
            self._trace_multi_collective_graph(tp_group="3", ep_group="ep")
        )
        fp_world_g3 = _graph_fingerprint(
            self._trace_multi_collective_graph(tp_group="7", ep_group="ep")
        )
        assert fp_world_g0 is not None
        assert fp_world_g0 == fp_world_g1 == fp_world_g3

        @torch.library.custom_op("test_fix_b::world_pallas_op", mutates_args=())
        def _world_pallas_op(
            x: torch.Tensor, device_assignment: list[int]
        ) -> torch.Tensor:
            return x.clone()

        @_world_pallas_op.register_fake
        def _(x: torch.Tensor, device_assignment: list[int]) -> torch.Tensor:
            return torch.empty_like(x)

        def _trace_with_world_pallas(tp_group: str) -> fx.GraphModule:
            def forward(x: torch.Tensor) -> torch.Tensor:
                y = torch.ops._c10d_functional.all_reduce(x, "sum", tp_group)
                w = torch.ops._c10d_functional.wait_tensor(y)
                return torch.ops.test_fix_b.world_pallas_op(w, [0, 1, 2, 3, 4, 5, 6, 7])

            return fx.symbolic_trace(forward)

        assert _graph_fingerprint(_trace_with_world_pallas("1")) == _graph_fingerprint(
            _trace_with_world_pallas("7")
        )

        monkeypatch.setattr(
            tpu_compiler_mod, "_supports_global_subgroup_tiling", lambda: True
        )
        assert (
            _graph_fingerprint(self._trace_all_reduce_graph("1"))
            == _graph_fingerprint(self._trace_all_reduce_graph("3"))
            == _graph_fingerprint(self._trace_all_reduce_graph("5"))
            == _graph_fingerprint(self._trace_all_reduce_graph("7"))
        )

    def test_large_constant_tensor_beyond_65536_numel_does_not_collide(self):
        """Non-Parameter get_attr constant tensors > 65536 elements are hashed
        in full while nn.Parameter weights hash by dtype and shape."""
        numel = tpu_compiler_mod._FINGERPRINT_MAX_NUMEL + 1
        assert numel == 65537

        const_a = torch.zeros(numel, dtype=torch.float32)
        const_b = torch.zeros(numel, dtype=torch.float32)
        const_b[-1] = 1.0

        class _LargeConstMod(torch.nn.Module):
            def __init__(self, const: torch.Tensor) -> None:
                super().__init__()
                self._tensor_constant0 = const

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self._tensor_constant0

        gm_const_a = fx.symbolic_trace(_LargeConstMod(const_a))
        gm_const_b = fx.symbolic_trace(_LargeConstMod(const_b))

        fp_const_a = _graph_fingerprint(gm_const_a)
        fp_const_b = _graph_fingerprint(gm_const_b)
        assert fp_const_a is not None
        assert fp_const_b is not None
        assert fp_const_a != fp_const_b

        class _LargeParamMod(torch.nn.Module):
            def __init__(self, weight: torch.Tensor) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.weight

        gm_param_a = fx.symbolic_trace(_LargeParamMod(const_a))
        gm_param_b = fx.symbolic_trace(_LargeParamMod(const_b))
        assert _graph_fingerprint(gm_param_a) == _graph_fingerprint(gm_param_b)

    def test_replay_from_peer_scans_all_candidate_paths_for_subgroup_peers(
        self, tmp_path, monkeypatch
    ):
        """In 8-rank 4xTP2 subgroup-bound mode, each follower skips earlier
        subgroups' artifacts in _peer_paths and replays its own subgroup
        leader's artifact."""
        _keep_alive = self._install_group_stand_ins(monkeypatch)
        monkeypatch.setattr(tpu_compiler_mod, "_distributed_world_size", lambda: 8)
        monkeypatch.setattr(
            tpu_compiler_mod, "_supports_global_subgroup_tiling", lambda: False
        )

        tls = threading.local()
        monkeypatch.setattr(
            tpu_compiler_mod,
            "_distributed_rank",
            lambda: getattr(tls, "rank", None),
        )

        compilers, compile, compiled, _replayed = self._peer_compilers(
            tmp_path, monkeypatch, n=8
        )
        rank_to_pg = {
            0: "1",
            1: "1",
            2: "3",
            3: "3",
            4: "5",
            5: "5",
            6: "7",
            7: "7",
        }

        def _compile_rank(r: int):
            tls.rank = r
            return compile(compilers[r], self._trace_all_reduce_graph(rank_to_pg[r]))

        _compile_rank(0)

        follower_threads: list[threading.Thread] = []
        results: dict[int, tuple] = {}
        followers_waiting = threading.Barrier(5)
        orig_peer_paths = tpu_compiler_mod.TpuCompilerAdaptor._peer_paths

        def _synced_peer_paths(adaptor, key):
            if getattr(tls, "first_poll", False):
                tls.first_poll = False
                followers_waiting.wait(timeout=5.0)
            return orig_peer_paths(adaptor, key)

        monkeypatch.setattr(
            tpu_compiler_mod.TpuCompilerAdaptor, "_peer_paths", _synced_peer_paths
        )

        def _run_waiter(r: int, is_follower: bool = False) -> None:
            tls.first_poll = is_follower
            with wait_for_peer_artifact(5.0):
                results[r] = _compile_rank(r)

        for r in (1, 3, 5, 7):
            t = threading.Thread(target=_run_waiter, args=(r, True))
            t.start()
            follower_threads.append(t)

        followers_waiting.wait(timeout=5.0)
        for r in (2, 4, 6):
            _run_waiter(r)

        for t in follower_threads:
            t.join(5.0)
            assert not t.is_alive()

        assert sorted(compiled) == [0, 2, 4, 6]
        for follower, leader in ((1, 0), (3, 2), (5, 4), (7, 6)):
            fn, (_, follower_path, _) = results[follower]
            assert fn() == f"entry-from-rank-{leader}"
            leader_cache_dir = compilers[leader].cache_dir
            assert leader_cache_dir is not None
            leader_path = os.path.join(leader_cache_dir, "artifact_bucket")
            assert os.path.samefile(follower_path, leader_path)


class _FakeBundledAOTEntry:
    def __init__(self, *, fail: bool = False):
        self.guards_expr = "guards"
        self.sanitized_aot_config = None
        self.fail = fail

    def wrap_post_compile(self, example_inputs, aot_config, options):
        if self.fail:
            raise RuntimeError("wrap_post_compile failed")
        return lambda boxed_args: sum(boxed_args)

    def __deepcopy__(self, memo):
        raise AssertionError("replay must not deepcopy the entry")


def test_deserialize_entry_does_not_copy_the_entry():
    """Replay rebuilds the runnable from the entry itself without leaving a
    __deepcopy__ override on the entry instance, even when deserialization
    raises."""
    import pickle

    from vllm_torchtpu.compilation.tpu_compiler import _deserialize_entry

    entry = _FakeBundledAOTEntry()
    assert _deserialize_entry(entry)(2, 3, 7) == 12
    assert "__deepcopy__" not in entry.__dict__
    assert isinstance(pickle.loads(pickle.dumps(entry)), _FakeBundledAOTEntry)

    failing_entry = _FakeBundledAOTEntry(fail=True)
    with pytest.raises(RuntimeError, match="wrap_post_compile failed"):
        _deserialize_entry(failing_entry)
    assert "__deepcopy__" not in failing_entry.__dict__
    assert isinstance(pickle.loads(pickle.dumps(failing_entry)), _FakeBundledAOTEntry)
