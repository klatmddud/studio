from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

__all__ = [
    "check_system",
    "check_gpu_driver",
    "check_cuda_toolkit",
    "check_cudnn",
    "check_env"
]


def print_header(title: str) -> None:
    print(f"\n[{title}]")


def print_kv(key: str, value: object) -> None:
    print(f"{key:<20}: {value}")


def run_command(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        return False, "command not found"
    except subprocess.TimeoutExpired:
        return False, "command timed out"
    except OSError as exc:
        return False, str(exc)

    output = completed.stdout.strip() or completed.stderr.strip() or "(no output)"
    return completed.returncode == 0, output


def first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text.strip() else "(empty)"


def last_line(text: str) -> str:
    return text.splitlines()[-1].strip() if text.strip() else "(empty)"


def format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


def print_env_vars(names: Iterable[str]) -> None:
    for name in names:
        value = os.environ.get(name, "(not set)")
        print_kv(name, value)


def find_executable(command: str) -> str | None:
    command_path = Path(command)
    if command_path.parent != Path("."):
        if command_path.is_file() and (os.name == "nt" or os.access(command_path, os.X_OK)):
            return str(command_path)
        return None

    search_path = os.environ.get("PATH") or os.defpath
    directories = [Path(entry.strip('"')) for entry in search_path.split(os.pathsep) if entry]

    if os.name == "nt":
        directories = [Path.cwd(), *directories]
        pathext = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep)
        suffixes = [""] if command_path.suffix else ["", *pathext]
    else:
        suffixes = [""]

    for directory in directories:
        for suffix in suffixes:
            candidate = directory / f"{command}{suffix}"
            if not candidate.is_file():
                continue
            if os.name == "nt" or os.access(candidate, os.X_OK):
                return str(candidate)

    return None


def find_cudnn_files(cuda_path: str) -> list[Path]:
    root = Path(cuda_path)
    search_dirs = [root, root / "bin", root / "lib", root / "lib64"]
    patterns = ("cudnn*.dll", "libcudnn*", "cudnn*.so*", "libcudnn*.dylib")

    candidates: list[Path] = []
    seen: set[Path] = set()
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            for path in search_dir.glob(pattern):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)
    return candidates


def check_system() -> None:
    print_header("System")
    print_kv("Python", sys.version.split()[0])
    print_kv("Executable", sys.executable)
    print_kv("Platform", platform.platform())


def check_gpu_driver() -> None:
    print_header("GPU Driver")

    nvidia_smi = find_executable("nvidia-smi")
    print_kv("nvidia-smi", nvidia_smi or "(not found)")

    if not nvidia_smi:
        return

    ok, output = run_command(
        [
            nvidia_smi,
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    print_kv("GPU query", "OK" if ok else "FAILED")
    if ok:
        for line in output.splitlines():
            print(f"  - {line.strip()}")
    else:
        print(f"  - {first_line(output)}")


def check_cuda_toolkit() -> None:
    print_header("CUDA")
    print_env_vars(["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"])

    nvcc = find_executable("nvcc")
    print_kv("nvcc", nvcc or "(not found)")

    if nvcc:
        ok, output = run_command([nvcc, "--version"])
        print_kv("nvcc version", last_line(output) if ok else f"FAILED: {first_line(output)}")

    if torch is None:
        print_kv("torch", "(not installed)")
        return

    print_kv("torch", torch.__version__)
    print_kv("torch CUDA build", torch.version.cuda or "(CPU build)")
    print_kv("CUDA built", torch.backends.cuda.is_built())
    print_kv("CUDA available", torch.cuda.is_available())

    if not torch.cuda.is_available():
        return

    print_kv("CUDA device count", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"  - GPU {index}: {props.name}")
        print(f"    capability      : {props.major}.{props.minor}")
        print(f"    total memory    : {format_bytes(props.total_memory)}")
        print(f"    multi processors: {props.multi_processor_count}")


def check_cudnn() -> None:
    print_header("cuDNN")
    print_env_vars(["CUDNN_PATH", "CUDNN_ROOT"])

    if torch is None:
        print_kv("torch", "(not installed)")
        return

    print_kv("cuDNN enabled", torch.backends.cudnn.enabled)
    print_kv("cuDNN available", torch.backends.cudnn.is_available())
    print_kv("cuDNN version", torch.backends.cudnn.version() or "(unknown)")

    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        cudnn_candidates = find_cudnn_files(cuda_path)
        if cudnn_candidates:
            print_kv("cuDNN files", cudnn_candidates[0])
            if len(cudnn_candidates) > 1:
                print(f"  - additional matches: {len(cudnn_candidates) - 1}")
        elif torch.backends.cudnn.is_available():
            print_kv("cuDNN files", "(not found under CUDA path; likely bundled with PyTorch)")
        else:
            print_kv("cuDNN files", "(not found under CUDA path)")


def check_env() -> None:
    print("Environment check for GPU / CUDA / cuDNN")
    check_system()
    check_gpu_driver()
    check_cuda_toolkit()
    check_cudnn()


if __name__ == "__main__":
    check_env()
