"""Read-only GPU occupancy guard. Never stops processes or resets devices."""
from __future__ import annotations

import os
import re
import subprocess

IDLE_MEMORY_MIB = 32
UUID_PATTERN = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"


def require(value, message):
    if not value:
        raise ValueError(message)


def occupancy(gpu_csv, process_csv, visible_uuid, own_pid, *, starting=False):
    """A launch needs two idle GPUs; each forward keeps one other GPU idle.

    Memory and compute processes are both checked. Utilization is not used.
    The checks derive from the completed study's gpu_spare_snapshot function.
    """
    require(isinstance(visible_uuid, str) and
            re.fullmatch(UUID_PATTERN, visible_uuid) is not None,
            "Set CUDA_VISIBLE_DEVICES to exactly one full GPU UUID.")
    require(type(own_pid) is int and own_pid > 0, "Invalid process identity.")
    devices, indices = {}, set()
    for line in gpu_csv.splitlines():
        if not line.strip():
            continue
        fields = [v.strip() for v in line.split(",")]
        require(len(fields) == 3 and fields[0].isdigit() and
                re.fullmatch(UUID_PATTERN, fields[1]) is not None
                and fields[2].isdigit(), "Invalid nvidia-smi GPU inventory.")
        index, uuid, memory = int(fields[0]), fields[1], int(fields[2])
        require(uuid not in devices and index not in indices, "Duplicate GPU inventory.")
        indices.add(index)
        devices[uuid] = {"memory_mib": memory, "processes": set()}
    require(devices and visible_uuid in devices, "Selected GPU UUID is absent from inventory.")
    for line in process_csv.splitlines():
        if not line.strip():
            continue
        fields = [v.strip() for v in line.split(",")]
        require(len(fields) == 2 and fields[0] in devices and fields[1].isdigit()
                and int(fields[1]) > 0, "Invalid nvidia-smi compute-process inventory.")
        devices[fields[0]]["processes"].add(int(fields[1]))
    allocated = devices[visible_uuid]
    require(allocated["processes"] <= {own_pid}, "Selected GPU has another process; choose a free GPU.")
    if starting:
        require(not allocated["processes"] and allocated["memory_mib"] <= IDLE_MEMORY_MIB,
                "Selected GPU is not idle before model execution.")
    else:
        require(allocated["memory_mib"] <= IDLE_MEMORY_MIB or own_pid in allocated["processes"],
                "Selected GPU has unattributed memory occupancy.")
    spare = sum(uuid != visible_uuid and value["memory_mib"] <= IDLE_MEMORY_MIB
                and not value["processes"] for uuid, value in devices.items())
    require(spare >= 1, "Keep at least one other GPU idle; no idle spare is available.")
    return {"other_idle_gpu_n": spare, "idle_memory_mib_maximum": IDLE_MEMORY_MIB,
            "memory_and_compute_processes_checked": True,
            "single_allocated_gpu_has_no_other_process": True}


class GPUChecker:
    def __init__(self, *, query=None, environ=None, own_pid=None):
        self.environ = os.environ if environ is None else environ
        self.own_pid = os.getpid() if own_pid is None else own_pid
        self.uuid = self.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.query = subprocess.check_output if query is None else query

    def check(self, *, starting=False):
        gpu_csv = self.query(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                              "--format=csv,noheader,nounits"], text=True, timeout=30)
        process_csv = self.query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                  "--format=csv,noheader,nounits"], text=True, timeout=30)
        return occupancy(gpu_csv, process_csv, self.uuid, self.own_pid, starting=starting)
