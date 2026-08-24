"""
test_convergence.py

Black-box correctness tests for the compiled sandpile_2d binary. These
don't test the MPI internals directly (that would need a C test
harness); instead they verify the property that actually matters:
the physical result must not depend on how many ranks you run with.

Run from the repo root, after `make`:
    python3 -m pytest tests/ -v
"""

import hashlib
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO_ROOT, "sandpile_2d")


def _run(np, size, out_file, steps=20000, maxh=4, seed=42):
    cmd = [
        "mpirun", "--allow-run-as-root", "--oversubscribe",
        "-np", str(np), BINARY,
        "--size", str(size), "--maxh", str(maxh),
        "--steps", str(steps), "--seed", str(seed),
        "--out", out_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"sandpile_2d failed: {result.stderr}"
    return result.stdout


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def require_binary():
    if not os.path.exists(BINARY):
        pytest.skip("sandpile_2d not built — run `make` first")


def test_binary_exists():
    assert os.path.exists(BINARY)


@pytest.mark.parametrize("np", [1, 2, 4, 8])
def test_converges_without_error(tmp_path, np):
    out_file = str(tmp_path / f"sand_{np}.ppm")
    stdout = _run(np, size=64, out_file=out_file)
    assert "Converged after" in stdout
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0


def test_decomposition_independence(tmp_path):
    """
    The single most important correctness property: the converged
    grid must be IDENTICAL regardless of how many MPI ranks were used
    to compute it. This catches halo-exchange bugs (wrong neighbour,
    off-by-one in the padded array, stale data from skipping a Wait)
    that would otherwise only show up as a subtly wrong-looking image.
    """
    hashes = {}
    for np in (1, 2, 4, 8):
        out_file = str(tmp_path / f"decomp_{np}.ppm")
        _run(np, size=64, out_file=out_file)
        hashes[np] = _md5(out_file)

    assert len(set(hashes.values())) == 1, (
        f"Converged grid differs between rank counts: {hashes} — "
        "this indicates a halo-exchange or boundary-condition bug."
    )


def test_larger_grid_still_decomposition_independent(tmp_path):
    hashes = {}
    for np in (4, 16):
        out_file = str(tmp_path / f"decomp128_{np}.ppm")
        _run(np, size=128, out_file=out_file, steps=20000)
        hashes[np] = _md5(out_file)
    assert len(set(hashes.values())) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
