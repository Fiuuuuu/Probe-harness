"""Docker availability and D10 hardening for SWE-bench eval containers."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import traceback
from typing import Any

DOCKER_HARDENING_CREATE_KWARGS: dict[str, Any] = {
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges:true"],
    "pids_limit": 512,
}

# CLI flags for documentation / subprocess wrappers that accept docker run argv.
DOCKER_HARDENING_RUN_ARGS: tuple[str, ...] = (
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "512",
)


def is_docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def install_secure_swebench_container_factory() -> None:
    """Monkeypatch swebench harness container creation to apply D10 hardening."""
    import docker.errors
    import swebench.harness.run_evaluation as run_eval
    from swebench.harness.docker_utils import cleanup_container
    from swebench.harness.run_evaluation import EvaluationError
    from swebench.harness.utils import TestSpec
    from swebench.image_builder.constants import CONTAINER_USER

    def create_container_d10(
        test_spec: TestSpec,
        client: Any,
        run_id: str,
        logger: logging.Logger,
    ):
        container = None
        try:
            try:
                client.images.get(test_spec.image)
            except docker.errors.ImageNotFound:
                try:
                    logger.info("Image not found locally, attempting to pull...")
                    client.images.pull(test_spec.image)
                except docker.errors.ImageNotFound as exc:
                    raise EvaluationError(
                        test_spec.instance_id,
                        f"Image {test_spec.image} not found for {test_spec.instance_id}",
                        logger,
                    ) from exc

            logger.info(f"Creating hardened container for {test_spec.instance_id}...")
            container_name = f"sweb.eval.{test_spec.instance_id.lower()}.{run_id}"
            try:
                old = client.containers.get(container_name)
                old.remove(force=True)
                logger.info(f"Removed existing container {container_name}")
            except docker.errors.NotFound:
                pass
            except Exception:
                pass

            create_kwargs = {
                "image": test_spec.image,
                "name": container_name,
                "user": CONTAINER_USER,
                "detach": True,
                "command": "tail -f /dev/null",
                **DOCKER_HARDENING_CREATE_KWARGS,
            }
            try:
                container = client.containers.create(**create_kwargs)
            except docker.errors.APIError as e:
                if "409" in str(e) or "Conflict" in str(e):
                    container_name = f"{container_name}.{int(time.time())}"
                    logger.info(f"Retrying with unique name: {container_name}")
                    create_kwargs["name"] = container_name
                    container = client.containers.create(**create_kwargs)
                else:
                    raise
            logger.info(f"Container for {test_spec.instance_id} created (D10): {container.id}")
            return container
        except Exception as e:
            logger.error(f"Error creating container for {test_spec.instance_id}: {e}")
            logger.info(traceback.format_exc())
            cleanup_container(client, container, logger)
            raise EvaluationError(test_spec.instance_id, str(e), logger) from e

    run_eval.create_container = create_container_d10  # type: ignore[method-assign]
