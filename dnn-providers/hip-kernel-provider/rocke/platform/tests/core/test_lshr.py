# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``arith.lshr`` and ``vector.lshr`` are logical shifts in the element's own width.

LLVM's ``lshr iW`` (and ``lshr <N x iW>``, lane by lane) shifts zeros in at
bit ``W - 1``. The HIP lowering reads the value, or each lane, through the
unsigned type of its width before shifting, so a negative i8 or i16 does not
shift its sign bits in and an i64 keeps its high word. Covers the HIP source,
that hipcc accepts it, and, when a GPU is present, that the LLVM route and
hipcc on the HIP source compute the same values as numpy, on negative and
positive values alike. The HIP source is the Python engine's, and must equal
the C++ engine's when the ``rocke_engine`` binding is built, so the HIP cases
run on lanes that do not build it.
"""

from __future__ import annotations

import ctypes
import shutil
import struct
import subprocess
from typing import NamedTuple

import numpy as np
import pytest
from rocke.core.arch.target import (
    base_arch_from_target_id,
    compiler_target_from_target_id,
)
from rocke.core.ir import F32, I8, I16, I32, I64, IRBuilder, PtrType, Type
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.tests.engines import (
    available_engines,
    lower_hip_with_available_engines,
)

_LANES = 64


class LshrCase(NamedTuple):
    """One shift under test: ``vector_lanes`` is ``None`` for a scalar
    ``arith.lshr``, else the lane count of a ``vector.lshr``."""

    dtype: Type
    vector_lanes: int | None
    unsigned: str  # the unsigned C type of the element's width
    numpy_dtype: type[np.signedinteger]

    @property
    def elements_per_thread(self) -> int:
        return self.vector_lanes or 1


CASES = (
    LshrCase(I8, None, "uint8_t", np.int8),
    LshrCase(I16, None, "uint16_t", np.int16),
    LshrCase(I32, None, "uint32_t", np.int32),
    LshrCase(I64, None, "uint64_t", np.int64),
    LshrCase(I8, 4, "uint8_t", np.int8),
    LshrCase(I16, 2, "uint16_t", np.int16),
    LshrCase(I32, 2, "uint32_t", np.int32),
)


def _id(case):
    if case.vector_lanes is None:
        return case.dtype.name
    return f"{case.vector_lanes}x{case.dtype.name}"


def _lshr_kernel(case):
    """Thread ``t`` shifts ``X`` right by ``S`` and stores the result to ``Y``:
    element ``t`` with a scalar ``arith.lshr``, or the ``vector_lanes``
    elements from ``t * vector_lanes`` on with one ``vector.lshr``."""
    b = IRBuilder(f"lshr_{_id(case)}")
    b.kernel.attrs["max_workgroup_size"] = _LANES
    if case.vector_lanes is None:
        x, s, y = (b.param(name, PtrType(case.dtype, "global")) for name in "XSY")
        thread = b.thread_id_x()
        value = b.global_load(x, thread, case.dtype)
        shift = b.global_load(s, thread, case.dtype)
        b.global_store(y, thread, b.lshr(value, shift))
        b.ret()
        return b.kernel
    x, s, y = (b.param(name, PtrType(I8, "global")) for name in "XSY")
    nbytes = case.vector_lanes * np.dtype(case.numpy_dtype).itemsize
    total = b.const_i32(_LANES * nbytes)
    offset = b.mul(b.thread_id_x(), b.const_i32(nbytes))
    zero = b.const_i32(0)
    value = b.buffer_load_vN(
        b.buffer_rsrc(x, total), offset, zero, case.dtype, case.vector_lanes
    )
    shift = b.buffer_load_vN(
        b.buffer_rsrc(s, total), offset, zero, case.dtype, case.vector_lanes
    )
    b.buffer_store_vN(
        b.buffer_rsrc(y, total), offset, zero, b.vector_lshr(value, shift)
    )
    b.ret()
    return b.kernel


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_hip_shifts_through_the_unsigned_type_of_the_width(case):
    hip = lower_hip_with_available_engines(_lshr_kernel(case), "gfx942")
    shifts = [line.strip() for line in hip.splitlines() if " >> " in line]
    assert len(shifts) == case.elements_per_thread
    assert all(f"(({case.unsigned})" in line for line in shifts), shifts


def test_hip_refuses_a_shift_with_no_unsigned_type_of_its_width():
    """An f32 ``arith.lshr`` has no unsigned type to read it through: the
    Python engine raises ``KeyError`` naming the type, as its other lookups in
    the unsigned-type table do, and the C++ engine fails with the same name."""
    b = IRBuilder("lshr_f32")
    y = b.param("Y", PtrType(F32, "global"))
    value = b.param("value", F32)
    b.global_store(y, b.const_i32(0), b.lshr(value, value))
    b.ret()
    with pytest.raises(KeyError) as python_error:
        lower_kernel_to_hip(b.kernel, arch="gfx942")
    assert python_error.value.args == ("f32",)
    if "cpp" in available_engines():
        import rocke_engine

        with pytest.raises(RuntimeError) as native_error:
            rocke_engine.lower_serialized_ir_hip(serialize(b.kernel), "gfx942")
        assert str(native_error.value).endswith(": f32"), str(native_error.value)


def _hipcc(kernel, target_id, out_dir):
    """Compile the HIP source emitted for ``kernel`` with hipcc; return the
    code object."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc not found")
    hip = lower_hip_with_available_engines(kernel, base_arch_from_target_id(target_id))
    source = out_dir / f"{kernel.name}.hip"
    blob = out_dir / f"{kernel.name}.hsaco"
    source.write_text(hip)
    subprocess.run(
        [hipcc, f"--offload-arch={compiler_target_from_target_id(target_id)}",
         "--genco", "-O3", str(source), "-o", str(blob)],
        check=True,
        capture_output=True,
        timeout=240,
    )  # fmt: skip
    return blob.read_bytes()


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_hip_source_compiles(tmp_path, arch):
    for case in CASES:
        assert _hipcc(_lshr_kernel(case), arch, tmp_path)


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _run(blob, kernel_name, x, s):
    """Launch one workgroup on ``x`` and ``s``; return ``Y`` as ``x``'s dtype."""
    from rocke.runtime.hip_module import Runtime

    runtime = Runtime()
    module = runtime.load_module(blob)
    out = np.zeros_like(x)
    buffers = [runtime.alloc(x.nbytes) for _ in range(3)]
    try:
        for device, host in zip(buffers, (x, s)):
            runtime.memcpy_h2d(
                device, (ctypes.c_ubyte * host.nbytes).from_buffer(host), host.nbytes
            )
        runtime.launch_blocking(
            module.get_function(kernel_name),
            (1, 1, 1),
            (_LANES, 1, 1),
            struct.pack("3Q", *buffers),
        )
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * out.nbytes).from_buffer(out), buffers[2], out.nbytes
        )
    finally:
        for device in buffers:
            runtime.free(device)
        module.unload()
    return out


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
@pytest.mark.parametrize("case", CASES, ids=_id)
def test_gpu_matches_a_logical_shift(tmp_path, case, route):
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    kernel = _lshr_kernel(case)
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel

        blob = compile_kernel(kernel, arch=arch).hsaco
    else:
        blob = _hipcc(kernel, arch, tmp_path)
    np_dtype = case.numpy_dtype
    bits = np.dtype(np_dtype).itemsize * 8
    unsigned_dtype = np.dtype(f"u{bits // 8}")
    rng = np.random.default_rng(11)
    count = _LANES * case.elements_per_thread
    raw = rng.integers(0, 256, count * bits // 8, dtype=np.uint8)
    x = raw.view(np_dtype).copy()
    x.view(unsigned_dtype)[0] = 1 << (bits - 1)  # the top bit alone, in every type
    assert (x < 0).any() and (x > 0).any()
    s = rng.integers(0, bits, x.size).astype(np_dtype)
    expected = (x.view(unsigned_dtype) >> s.astype(unsigned_dtype)).view(np_dtype)
    np.testing.assert_array_equal(_run(blob, kernel.name, x, s), expected)
