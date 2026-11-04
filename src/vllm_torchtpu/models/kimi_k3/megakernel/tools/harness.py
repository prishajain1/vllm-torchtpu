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

"""Launching and bringing up a megakernel program without vLLM.

:func:`launch` (on the login node) re-runs the tool as one process per TPU
device of an existing allocation, 8 per host, the way vLLM's workers are laid
out. In each process :func:`init` reproduces the environment the Ray executor
gives a worker (``TORCH_TPU_SLICEBUILDER_ADDRESSES``, ``TORCH_TPU_TOPOLOGY``,
torchrun-style ranks) and opens the ``tpu_dist`` process group; programs then
build their ops exactly as they do inside the server.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

_Dynamic = Any
_CONST_4 = 4


T0 = time.perf_counter()
REPO = Path(__file__).resolve().parents[6]
LOCAL_WORLD = 8


def is_worker() -> bool:
    """Execute is worker logic."""
    return os.environ.get("RANK") is not None


def rank() -> int:
    """Execute rank logic."""
    return int(os.environ.get("RANK", "0"))


def host() -> int:
    """Execute host logic."""
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "8") or "8")
    return int(os.environ.get("RANK", "0")) // max(1, local_world)


def log_host(msg: str) -> None:
    """A line from every process, tagged with its host (host-side tools)."""
    sys.stdout.write(f"[{time.perf_counter() - T0:7.1f}s host {host()}] {msg}\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    """Execute log logic."""
    if rank() == 0:
        sys.stdout.write(f"[{time.perf_counter() - T0:7.1f}s r{rank():02d}] {msg}\n")
        sys.stdout.flush()


def launch(
    argv: list[str],
    jobid: str,
    venv: str,
    extra_env: dict[str, str],
    per_host: int = LOCAL_WORLD,
) -> int:
    """Run ``python -m . argv`` on the four hosts of ``jobid``,.

    ``per_host`` processes each (default one per TPU device).
    """
    _sp_run = vars(subprocess)["run"]
    nodelist = _sp_run(
        ["/usr/bin/squeue", "-j", jobid, "-h", "-o", "%N"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not nodelist:
        _err_msg = f"job {jobid} is not running"
        raise SystemExit(_err_msg)
    hosts = _sp_run(
        ["/usr/bin/scontrol", "show", "hostnames", nodelist],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    if len(hosts) != _CONST_4:
        _err_msg = (
            f"job {jobid} holds {len(hosts)} hosts; the K3 programs need 4 (32 devices)"
        )
        raise SystemExit(_err_msg)
    ips = [
        _sp_run(
            ["/usr/bin/getent", "hosts", h], capture_output=True, text=True, check=False
        ).stdout.split()[0]
        for h in hosts
    ]
    env = dict(os.environ)
    env.update(extra_env)
    env.update(
        {
            "LOCAL_WORLD_SIZE": str(per_host),
            "TPU_NUM_HOSTS": str(len(hosts)),
            "PYTHONPATH": f"{REPO / 'src'}:{REPO}:{env.get('PYTHONPATH', '')}",
            "MASTER_PORT": env.get(
                "MASTER_PORT", str(20000 + secrets.randbelow(20000))
            ),
            "HF_HUB_OFFLINE": "1",
            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", ""),
            "TORCH_TPU_TOPOLOGY": "2,2,4,2",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
                f"{ip}:{8070 + i}" for ip in ips for i in range(per_host)
            ),
            "TORCH_TPU_XPROF_SESSION_ID": str(time.time_ns()),
        }
    )
    command = [
        "/usr/bin/srun",
        f"--jobid={jobid}",
        "--overlap",
        "--nodes=4",
        f"--ntasks={4 * per_host}",
        f"--ntasks-per-node={per_host}",
        "--kill-on-bad-exit=1",
        "--unbuffered",
        "--export=ALL",
        f"--chdir={REPO}",
        "bash",
        "-c",
        (
            "RANK=${SLURM_PROCID:-0} LOCAL_RANK=${SLURM_LOCALID:-0} "
            "LOCAL_WORLD_SIZE=${SLURM_NTASKS_PER_NODE:-${LOCAL_WORLD_SIZE:-8}} "
            'WORLD_SIZE=${SLURM_NTASKS:-1} exec "$@"'
        ),
        "bash",
        f"{venv}/bin/python",
        "-m",
        "vllm_torchtpu.models.kimi_k3.megakernel.tools",
        *argv,
    ]
    return vars(subprocess)["call"](command, env=env)


def init() -> None:
    """Bring up torch_tpu and the ``tpu_dist`` process group in a worker."""
    world_size = int(os.environ.get("WORLD_SIZE", "32"))
    slice_addrs = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "")
    host_ips = (
        [a.split(":")[0] for a in slice_addrs.split(",") if a]
        if slice_addrs
        else ["127.0.0.1"]
    )
    master_addr = host_ips[0] if host_ips else "127.0.0.1"
    master_port = os.environ.get("MASTER_PORT", "29500")
    os.environ.setdefault(
        "LOCAL_RANK", str(int(os.environ.get("RANK", "0")) % LOCAL_WORLD)
    )
    os.environ.setdefault("LOCAL_WORLD_SIZE", str(LOCAL_WORLD))
    os.environ.setdefault("TPU_NUM_HOSTS", str(len(set(host_ips))))
    os.environ.setdefault("TORCH_TPU_TOPOLOGY", "2,2,4,2")

    log("torch_tpu imported")
    torch.distributed.init_process_group(
        backend="tpu_dist",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank(),
    )
    log("tpu_dist process group up")


def sync() -> None:
    """Execute sync logic."""
    torch.distributed.barrier()


def git_revision() -> str:
    """Execute git revision logic."""
    _sp_run = vars(subprocess)["run"]
    out = _sp_run(
        ["/usr/bin/git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = _sp_run(
        [
            "/usr/bin/git",
            "-C",
            str(REPO),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return out.stdout.strip() + ("+dirty" if dirty else "")


class Records:
    """One JSON record per result (rank 0), with what produced it."""

    def __init__(
        self, path: str | None, command: str, config: _Dynamic, jobid: str = ""
    ) -> None:
        """Execute init logic."""
        self.path, self.command = path, command
        self.context = {
            "git": git_revision(),
            "config": config.key(),
            "jobid": jobid,
        }

    def add(self, **fields: _Dynamic) -> None:
        """Execute add logic."""
        if rank() != 0 or not self.path:
            return
        record = {
            "command": self.command,
            **fields,
            **self.context,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with Path(self.path).open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def fail(message: str) -> None:
    """Execute fail logic."""
    log(f"FAIL: {message}")
    sys.exit(1)
