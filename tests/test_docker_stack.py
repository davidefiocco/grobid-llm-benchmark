"""Stack sanity checks. Marked ``docker``; skipped when no docker daemon is present.

``compose config`` lints the stack for every profile combination (cheap). The optional
crf-lite service boot (guarded by RUN_GROBID_BOOT=1) pulls a ~477 MB multi-arch image and
asserts /api/isalive.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.docker

_DEPLOY = Path(__file__).parent.parent / "deploy"


def _have_docker() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


requires_docker = pytest.mark.skipif(not _have_docker(), reason="docker daemon not available")


@requires_docker
@pytest.mark.parametrize(
    "profiles",
    [[], ["--profile", "dl-parity", "--profile", "glutton-local", "--profile", "ollama"]],
)
def test_compose_config_lints(profiles):
    env = {**os.environ, "GLUTTON_URL": "http://glutton:8080"}
    r = subprocess.run(
        ["docker", "compose", "--env-file", ".env.example", *profiles, "config"],
        cwd=_DEPLOY,
        capture_output=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr.decode()


@requires_docker
@pytest.mark.skipif(os.environ.get("RUN_GROBID_BOOT") != "1", reason="set RUN_GROBID_BOOT=1")
def test_crf_lite_service_is_alive():
    import httpx

    up = subprocess.run(
        ["docker", "compose", "--env-file", ".env.example", "up", "-d", "grobid"],
        cwd=_DEPLOY,
        capture_output=True,
    )
    assert up.returncode == 0, up.stderr.decode()
    try:
        alive = False
        for _ in range(40):
            try:
                resp = httpx.get("http://localhost:8070/api/isalive", timeout=5)
                if resp.status_code == 200 and "true" in resp.text.lower():
                    alive = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        assert alive, "GROBID crf-lite service did not report alive"
    finally:
        subprocess.run(["docker", "compose", "down"], cwd=_DEPLOY, capture_output=True)
