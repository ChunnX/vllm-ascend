# SPDX-License-Identifier: Apache-2.0
"""Record installed extension identity; this does not prove kernel provenance."""

import hashlib
import importlib.metadata
import os
import sys
from pathlib import Path


def main():
    print("python:", sys.executable)
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "pytest", "numpy"):
        try:
            print(package, importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            print(package, "not installed as a distribution")
    for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_CUSTOM_OPP_PATH", "SOC_VERSION"):
        print(key, os.getenv(key, "<unset>"))

    import torch
    import torch_npu
    import vllm

    import vllm_ascend
    from vllm_ascend.utils import enable_custom_op

    for module in (torch, torch_npu, vllm, vllm_ascend):
        print(module.__name__, module.__file__)
    if not enable_custom_op():
        raise RuntimeError("Ascend extension could not be enabled")
    import vllm_ascend.vllm_ascend_C as extension

    path = Path(extension.__file__).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print("extension:", path, "sha256:", digest)
    print("effective ASCEND_CUSTOM_OPP_PATH:", os.getenv("ASCEND_CUSTOM_OPP_PATH"))
    for name in ("npu_dcut_causal_conv1d", "npu_dcut_recurrent_gated_delta_rule"):
        op = getattr(torch.ops._C_ascend, name, None)
        if op is None:
            raise RuntimeError(f"Missing registered interface: {name}")
        print(name, op.default._schema)
    print("visible devices:", torch.npu.device_count())
    print("device:", torch.npu.get_device_name(0))
    print("NOTE: extension identity and schemas do not prove which tiling/kernel binary executes.")


if __name__ == "__main__":
    main()
