# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
import subprocess
import types
from importlib.metadata import version
from importlib.util import find_spec

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)


def _configure_triton_ptxas_for_new_gpus() -> None:
    """Use the system ptxas for architectures newer than bundled Triton."""
    if os.environ.get("TRITON_PTXAS_PATH"):
        return

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    candidates = (
        os.path.join(cuda_home, "bin", "ptxas"),
        "/usr/local/cuda/bin/ptxas",
        shutil.which("ptxas"),
    )
    system_ptxas = next(
        (
            path
            for path in candidates
            if path and os.path.isfile(path) and os.access(path, os.X_OK)
        ),
        None,
    )
    if system_ptxas is None:
        return

    try:
        from triton.backends import backends

        nvidia_backend = backends.get("nvidia")
        if nvidia_backend is None or nvidia_backend.driver is None:
            return
        if not nvidia_backend.driver.is_active():
            return

        target = nvidia_backend.driver().get_current_target()
        if target.arch < 110:
            return

        result = subprocess.run(
            [system_ptxas, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return

        os.environ["TRITON_PTXAS_PATH"] = system_ptxas
        logger.info(
            "Using system ptxas %s for Triton GPU architecture %s.",
            system_ptxas,
            target.arch,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("Cannot configure system ptxas for Triton: %s", exc)
    except Exception as exc:
        # Triton driver APIs vary across versions; this must remain best-effort.
        logger.debug("Failed to detect Triton GPU architecture: %s", exc)


_configure_triton_ptxas_for_new_gpus()

HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)
if HAS_TRITON:
    try:
        from triton.backends import backends

        # It's generally expected that x.driver exists and has
        # an is_active method.
        # The `x.driver and` check adds a small layer of safety.
        # The "cpu" backend's driver reports itself active unconditionally, so
        # it must be excluded here or every GPU host looks like it has 2 active
        # drivers. Triton's own driver selection skips it for the same reason
        # (see triton.runtime.driver._create_driver): CPU is only ever used
        # when explicitly selected.
        active_drivers = [
            x.driver
            for name, x in backends.items()
            if name != "cpu" and x.driver and x.driver.is_active()
        ]

        # Check Triton CPU
        if "cpu" in version("vllm"):
            if "cpu" in backends:
                HAS_TRITON = True
                # Suppress following warnings on CPU-only platforms
                if len(active_drivers) == 0:
                    active_drivers.append(backends["cpu"].driver)
            else:
                logger.warning(
                    "Triton is installed, but doesn't include CPU backend. "
                    "Disabling Triton."
                )
                HAS_TRITON = False

        # Check if we're in a distributed environment where CUDA_VISIBLE_DEVICES
        # or HIP_VISIBLE_DEVICES might be temporarily empty (e.g., Ray sets it to ""
        # during actor init)
        visible_devices_env = (
            "HIP_VISIBLE_DEVICES"
            if current_platform.is_rocm()
            else "CUDA_VISIBLE_DEVICES"
        )
        visible_devices = os.environ.get(visible_devices_env)
        is_distributed_env = (
            visible_devices is not None and len(visible_devices.strip()) == 0
        )

        # Apply lenient driver check for distributed environments
        if is_distributed_env and len(active_drivers) == 0:
            # Allow 0 drivers in distributed environments - they may become
            # active later when CUDA context is properly initialized
            logger.debug(
                "Triton found 0 active drivers in distributed environment. "
                "This is expected during initialization."
            )
        elif not is_distributed_env and len(active_drivers) != 1:
            # Strict check for non-distributed environments
            logger.info(
                "Triton is installed but %d active driver(s) found "
                "(expected 1). Disabling Triton to prevent runtime errors.",
                len(active_drivers),
            )
            HAS_TRITON = False
    except ImportError:
        # This can occur if Triton is partially installed or triton.backends
        # is missing.
        logger.warning(
            "Triton is installed, but `triton.backends` could not be imported. "
            "Disabling Triton."
        )
        HAS_TRITON = False
    except Exception as e:
        # Catch any other unexpected errors during the check.
        logger.warning(
            "An unexpected error occurred while checking Triton active drivers:"
            " %s. Disabling Triton.",
            e,
        )
        HAS_TRITON = False

if not HAS_TRITON:
    logger.info(
        "Triton not installed or not compatible; certain GPU-related"
        " functions will not be available."
    )


class TritonPlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.__version__ = "3.4.0"
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")
        self.Config = self._dummy_decorator("Config")
        self.cdiv = cdiv
        self.language = TritonLanguagePlaceholder()

    def _dummy_decorator(self, name):
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

        return decorator


class TritonLanguagePlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = lambda value: value
        self.dtype = None
        self.int64 = None
        self.int32 = None
        self.tensor = None
        self.exp = None
        self.exp2 = None
        self.log = None
        self.log2 = None
