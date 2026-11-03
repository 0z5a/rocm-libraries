# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CMake contracts for standalone platform and embedded rocKE test assets."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def platform() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "CMakeLists.txt").is_file():
        pytest.skip("source CMake project is absent from the installed test artifact")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for the source configuration contract")
    return root


@pytest.fixture
def rocke(platform: Path) -> Path:
    root = platform.parent
    if not (root / "CMakeLists.txt").is_file():
        pytest.skip("rocKE umbrella project is absent from this platform copy")
    return root


def _configure(
    source: Path, build: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DBUILD_TESTING=OFF",
            "-DROCKE_BUILD_PYENV=OFF",
            "-DROCKE_INSTALL_TESTS=ON",
            "-DROCKE_INSTALL_PYTHON_TESTS=ON",
            "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
            *options,
        ],
        capture_output=True,
        text=True,
    )


def _configure_embedded(
    rocke: Path, build: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    embedding = build.parent / "embedding source"
    embedding.mkdir(parents=True, exist_ok=True)
    (embedding / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25.2)\n"
        "project(rocke_embedding_contract LANGUAGES C CXX)\n"
        f'add_subdirectory("{rocke.as_posix()}" rocke)\n'
    )
    return _configure(
        embedding,
        build,
        "-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR=hip_kernel_provider",
        *options,
    )


def _external_asset_options(assets: Path) -> tuple[list[str], Path, Path, Path]:
    template = assets / "template source.json"
    bundles = assets / "bundle source"
    headers = assets / "pack headers"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n")

    eligible = bundles / "case" / "SdpaFwdSmoke" / "nested" / "graph.json"
    eligible.parent.mkdir(parents=True)
    eligible.write_text("{}\n")
    (eligible.parent / "meta.json").write_text("{}\n")
    default = bundles / "case" / "SdpaFwdSmoke" / "Default" / "graph.json"
    default.parent.mkdir(parents=True)
    default.write_text("{}\n")
    other = bundles / "case" / "OtherOp" / "graph.json"
    other.parent.mkdir(parents=True)
    other.write_text("{}\n")

    headers.mkdir(parents=True)
    (headers / "Gfx950AttentionDenseGeometry.hpp").write_text("// dense\n")
    (headers / "Gfx950AttentionUnified3DGeometry.hpp").write_text("// unified\n")

    options = [
        "-DROCKE_REQUIRE_EXTERNAL_TEST_ASSETS=ON",
        f"-DROCKE_HIPDNN_SDPA_TEMPLATE={template}",
        f"-DROCKE_HIPDNN_SDPA_BUNDLES={bundles}",
        f"-DROCKE_PROVIDER_PACK_HEADERS={headers}",
    ]
    return options, template, bundles, headers


def test_standalone_platform_copy_configures_with_paths_containing_spaces(
    platform: Path, tmp_path: Path
) -> None:
    copied_platform = tmp_path / "standalone source" / "platform"
    shutil.copytree(
        platform,
        copied_platform,
        ignore=shutil.ignore_patterns("__pycache__", ".venv*", "build*"),
    )

    result = _configure(copied_platform, tmp_path / "build directory")

    assert result.returncode == 0, result.stdout + result.stderr


def test_platform_cmake_has_no_embedding_repository_layout(platform: Path) -> None:
    cmake = (platform / "CMakeLists.txt").read_text()

    assert "projects/hipdnn" not in cmake
    assert "dnn-" + "providers/" not in cmake
    assert "Gfx950AttentionDenseGeometry.hpp" not in cmake
    assert "Gfx950AttentionUnified3DGeometry.hpp" not in cmake


@pytest.mark.parametrize("missing", ("template", "graphs", "header"))
def test_strict_embedded_configure_rejects_each_incomplete_asset_class(
    rocke: Path, tmp_path: Path, missing: str
) -> None:
    options, template, bundles, headers = _external_asset_options(
        tmp_path / "external assets"
    )
    if missing == "template":
        template.unlink()
    elif missing == "graphs":
        shutil.rmtree(bundles)
        bundles.mkdir()
    else:
        (headers / "Gfx950AttentionDenseGeometry.hpp").unlink()

    result = _configure_embedded(rocke, tmp_path / "build directory", *options)

    assert result.returncode != 0
    assert "required external test assets are incomplete" in (
        result.stdout + result.stderr
    )


@pytest.mark.parametrize(
    "disabled", ("ROCKE_INSTALL_TESTS", "ROCKE_INSTALL_PYTHON_TESTS")
)
def test_strict_embedded_configure_rejects_disabled_test_installation(
    rocke: Path, tmp_path: Path, disabled: str
) -> None:
    options, _, _, _ = _external_asset_options(tmp_path / "external assets")

    result = _configure_embedded(
        rocke,
        tmp_path / "build directory",
        *options,
        f"-D{disabled}=OFF",
    )

    assert result.returncode != 0
    output = " ".join((result.stdout + result.stderr).split())
    assert "require ROCKE_INSTALL_TESTS and ROCKE_INSTALL_PYTHON_TESTS" in output


def test_embedded_install_stages_assets_by_observable_contract(
    rocke: Path, tmp_path: Path
) -> None:
    options, _, _, _ = _external_asset_options(tmp_path / "external assets")
    build = tmp_path / "build directory"
    result = _configure_embedded(rocke, build, *options)
    assert result.returncode == 0, result.stdout + result.stderr

    install = tmp_path / "install prefix"
    installed = subprocess.run(
        [
            "cmake",
            "--install",
            str(build),
            "--prefix",
            str(install),
            "--config",
            "Release",
            "--component",
            "rocke_external_test_assets",
        ],
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr

    tests = install / "bin" / "hip_kernel_provider" / "tests" / "library" / "tests"
    hipdnn = tests / "sdpa" / "hipdnn"
    assert (hipdnn / "attention_sdpa_fwd.graph.json").is_file()
    graphs = hipdnn / "integration-test-bundles"
    assert [path.relative_to(graphs).as_posix() for path in graphs.rglob("*.json")] == [
        "case/SdpaFwdSmoke/nested/graph.json"
    ]
    packs = tests / "hkp_packs"
    assert sorted(path.name for path in packs.iterdir()) == [
        "Gfx950AttentionDenseGeometry.hpp",
        "Gfx950AttentionUnified3DGeometry.hpp",
    ]

    manifests = list(build.rglob("CTestTestfile.cmake.install"))
    assert manifests
    assert any(
        "--rocke-require-hipdnn-assets" in manifest.read_text()
        for manifest in manifests
    )
