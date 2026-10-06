"""Runtime launcher checks, independent of API credentials and PostgreSQL."""
import shutil
import subprocess


class DeploymentConfigurationError(RuntimeError):
    """The deployed environment cannot launch a required executable."""


def require_executable(server: str, command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise DeploymentConfigurationError(
            f"Server '{server}': required executable '{command}' is missing or "
            "not on PATH. Install the deployment requirements and check PATH."
        )
    return executable


def verify_launchers():
    for command in ("uv", "uvx"):
        executable = require_executable("deployment runtime", command)
        try:
            result = subprocess.run(
                [executable, "--version"], check=True, capture_output=True,
                text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            raise DeploymentConfigurationError(
                f"Server 'deployment runtime': executable '{command}' "
                "failed its bounded version check."
            ) from None
        print(result.stdout.strip())


if __name__ == "__main__":
    verify_launchers()
