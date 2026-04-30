#!/usr/bin/env python3
"""Install NVIDIA Nsight Systems to ~/nsight-systems (no sudo needed)."""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

INSTALLER_URL = (
    "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/"
    "2026_1/NsightSystems-linux-public-2026.1.2.63-3729663.run"
)
INSTALL_DIR = Path.home() / "nsight-systems"
INSTALLER_PATH = Path.home() / "nsys_installer.run"
BASHRC = Path.home() / ".bashrc"


def run(cmd, **kwargs):
    print(f"$ {' '.join(map(str, cmd))}")
    return subprocess.run(cmd, check=True, **kwargs)


def find_nsys(root: Path) -> Path | None:
    """Find the nsys binary inside the install directory."""
    for path in root.rglob("nsys"):
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Installer already downloaded at {dest}, skipping.")
        return
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  {pct:3d}%  {mb:6.1f} / {total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def install() -> None:
    INSTALLER_PATH.chmod(0o755)
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    # Try non-interactive flags first; fall back to interactive if it fails.
    try:
        run([
            str(INSTALLER_PATH),
            "--quiet",
            "--accept",
            f"--installdir={INSTALL_DIR}",
        ])
    except subprocess.CalledProcessError:
        print("\nNon-interactive install failed; running installer interactively.")
        print(f"When asked, set the install directory to: {INSTALL_DIR}\n")
        run([str(INSTALLER_PATH)])


def update_bashrc(bin_dir: Path) -> None:
    line = f'export PATH="{bin_dir}:$PATH"'
    if BASHRC.exists() and line in BASHRC.read_text():
        print(f"PATH entry already in {BASHRC}, skipping.")
        return
    with BASHRC.open("a") as f:
        f.write(f"\n# Added by install_nsys.py\n{line}\n")
    print(f"Appended PATH entry to {BASHRC}")


def main() -> int:
    # Skip if already installed and on PATH.
    existing = shutil.which("nsys")
    if existing:
        print(f"nsys already on PATH at {existing}")
        run(["nsys", "--version"])
        return 0

    download(INSTALLER_URL, INSTALLER_PATH)
    install()

    nsys = find_nsys(INSTALL_DIR)
    if nsys is None:
        print(f"ERROR: could not find nsys binary under {INSTALL_DIR}", file=sys.stderr)
        return 1

    print(f"\nFound nsys at: {nsys}")
    bin_dir = nsys.parent
    update_bashrc(bin_dir)

    # Verify in a fresh-ish env.
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    print()
    subprocess.run([str(nsys), "--version"], env=env, check=False)

    print("\nDone. Run `source ~/.bashrc` (or open a new shell) to pick up the PATH change.")
    print(f"Then: nsys profile -o test --trace=cuda,nvtx python your_script.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())