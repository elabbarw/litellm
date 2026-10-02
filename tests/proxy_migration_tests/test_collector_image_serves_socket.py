"""Image-level regression net for the collector's unix socket directory.

The collector creates its unix socket under ``/var/run/litellm`` at startup
(``SpendEventConsumer.serve`` mkdirs the parent before binding), so the
non-root image must let it do so with no mount at all, with a fresh named
volume (which inherits the image dir's ownership and mode), and as an
arbitrary uid in GID 0, the shape OpenShift restricted-v2 assigns. A
``PermissionError`` mkdir'ing the dir kills the process before it ever
listens, which is exactly what a helm install without the socket emptyDir
hits.

Gated on LITELLM_IMAGE (the tag of the image to exercise) so it is skipped in
the normal unit-test run and exercised only where an image has been built (the
image-scan workflow). Requires a working docker CLI.
"""

import os
import shutil
import subprocess
import time
import uuid

import pytest

IMAGE = os.getenv("LITELLM_IMAGE")
POSTGRES_IMAGE = os.getenv("LITELLM_TEST_POSTGRES_IMAGE", "postgres:16-alpine")
NON_ROOT_UID = "12345:0"
STARTUP_TIMEOUT_SECONDS = int(os.getenv("LITELLM_COMPONENT_STARTUP_TIMEOUT", "180"))

pytestmark = [
    pytest.mark.skipif(IMAGE is None, reason="requires a built image (set LITELLM_IMAGE)"),
    pytest.mark.skipif(shutil.which("docker") is None, reason="requires the docker CLI"),
]


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def _wait_until_postgres_ready(pg: str, attempts: int = 60) -> None:
    for _ in range(attempts):
        running = _docker(
            "ps",
            "--filter",
            f"name={pg}",
            "--filter",
            "status=running",
            "--format",
            "{{.Names}}",
            check=False,
        ).stdout
        if pg not in running:
            logs = _docker("logs", pg, check=False)
            pytest.fail(f"postgres container is not running:\n{logs.stdout}\n{logs.stderr}")
        ready = _docker("exec", pg, "pg_isready", "-U", "postgres", "-d", "litellm", check=False)
        if ready.returncode == 0:
            return
        time.sleep(1)
    pytest.fail(f"postgres never became ready after {attempts}s")


def _container_logs(container: str) -> str:
    logs = _docker("logs", container, check=False)
    return f"stdout:\n{logs.stdout}\nstderr:\n{logs.stderr}"


def _is_running(container: str) -> bool:
    return bool(
        _docker(
            "ps",
            "--filter",
            f"name={container}",
            "--filter",
            "status=running",
            "--format",
            "{{.Names}}",
            check=False,
        ).stdout.strip()
    )


@pytest.fixture()
def collector_stack(request):
    """A collector container and a fresh Postgres on a dedicated network.

    ``request.param`` is (user, mount_socket_volume): ``None`` runs the image's
    own USER, ``12345:0`` is the arbitrary uid in GID 0 OpenShift assigns. The
    collector runs the proxy startup lifespan before it binds its socket, so it
    needs a live database the way the component pods do; the socket itself must
    appear without any mounted dir. The schema belongs to the migrations Job,
    same as in the helm topology, so it is applied synchronously first and the
    collector boots with schema update off. Yields the collector container
    name. The containers, network and volume are torn down afterwards.
    """
    user, mount_socket_volume = request.param
    run_id = f"collectorsock-{uuid.uuid4().hex[:8]}"
    network = f"{run_id}-net"
    pg = f"{run_id}-pg"
    collector = f"{run_id}-app"
    volume = f"{run_id}-sock"

    _docker("network", "create", network)
    try:
        _docker(
            "run",
            "-d",
            "--name",
            pg,
            "--network",
            network,
            "-e",
            "POSTGRES_PASSWORD=pw",
            "-e",
            "POSTGRES_DB=litellm",
            POSTGRES_IMAGE,
        )
        _wait_until_postgres_ready(pg)
        assert IMAGE is not None
        _docker(
            "run",
            "--rm",
            "--name",
            f"{run_id}-migrations",
            "--network",
            network,
            "-e",
            f"DATABASE_URL=postgresql://postgres:pw@{pg}:5432/litellm",
            IMAGE,
            "migrations",
        )
        run_args = [
            "run",
            "-d",
            "--name",
            collector,
            "--network",
            network,
            "-e",
            f"DATABASE_URL=postgresql://postgres:pw@{pg}:5432/litellm",
            "-e",
            "LITELLM_MASTER_KEY=sk-collector-socket-test",
            "-e",
            "LITELLM_LOCAL_MODEL_COST_MAP=True",
            "-e",
            "DISABLE_SCHEMA_UPDATE=true",
        ]
        if user is not None:
            run_args += ["--user", user]
        if mount_socket_volume:
            run_args += ["-v", f"{volume}:/var/run/litellm"]
        _docker(*run_args, IMAGE, "collector")
        yield collector
    finally:
        _docker("rm", "-f", collector, check=False)
        _docker("rm", "-f", pg, check=False)
        _docker("volume", "rm", "-f", volume, check=False)
        _docker("network", "rm", network, check=False)


@pytest.mark.parametrize(
    "collector_stack",
    [
        pytest.param((None, False), id="default-user-no-mount"),
        pytest.param((None, True), id="default-user-named-volume"),
        pytest.param((NON_ROOT_UID, False), id="arbitrary-uid-gid0-no-mount"),
    ],
    indirect=True,
)
def test_collector_creates_its_socket_as_the_image_user(collector_stack: str) -> None:
    """The collector stays up and its socket exists under every supported runtime shape.

    On an image without /var/run/litellm baked in, the no-mount case dies on
    ``PermissionError: [Errno 13] Permission denied: '/var/run/litellm'`` and a
    fresh named volume inherits the same root-owned dir; the socket only appears
    when the image ships the dir owned 65532:0 and group-writable.
    """
    collector = collector_stack

    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    socket_seen = False
    while time.time() < deadline:
        if not _is_running(collector):
            pytest.fail(f"the collector exited before creating its socket.\n{_container_logs(collector)}")
        probe = _docker(
            "exec",
            collector,
            "sh",
            "-c",
            "test -S /var/run/litellm/collector.sock",
            check=False,
        )
        if probe.returncode == 0:
            socket_seen = True
            break
        time.sleep(2)

    inspect = _docker("inspect", "-f", "{{.State.Running}}", collector, check=False).stdout.strip()
    assert inspect == "true", (
        f"the collector container is not running (inspect State.Running={inspect!r}).\n{_container_logs(collector)}"
    )
    assert socket_seen, (
        f"/var/run/litellm/collector.sock never appeared within {STARTUP_TIMEOUT_SECONDS}s.\n"
        f"{_container_logs(collector)}"
    )
