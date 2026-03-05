#!/usr/bin/env python3
"""
RAM disk management for SPSA tuning workers.

Shared helpers for creating/removing ImDisk RAM disks on Windows and /dev/shm
on Linux. Used by worker.py and as a standalone CLI tool.

Usage:
    python ramdisk.py create --engine PATH [--ref PATH] [--concurrency N] [--drive X:] [--size MB] [--factor F]
    python ramdisk.py remove [--drive X:]
    python ramdisk.py clean  [--drive X:]
    python ramdisk.py status
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

logger = logging.getLogger("ramdisk")

IMDISK_URL = "https://sourceforge.net/projects/imdisk-toolkit/files/20250206/ImDiskTk-x64.zip/download"
RAMDISK_MARKER = "spsa_ramdisk"  # marker file to identify our RAM disks


# --- ImDisk helpers ---

def has_imdisk() -> bool:
    """Check if imdisk is available on the system."""
    return shutil.which("imdisk") is not None


def install_imdisk() -> bool:
    """Download and silently install ImDisk Toolkit. Returns True on success."""
    logger.info("Downloading ImDisk Toolkit from SourceForge...")
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="imdisk_")
        zip_path = os.path.join(tmp, "ImDiskTk-x64.zip")
        import ssl
        ctx = ssl.create_default_context()
        try:
            certifi = __import__("certifi")
            ctx.load_verify_locations(certifi.where())
        except (ImportError, ssl.SSLError):
            pass
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        try:
            with opener.open(IMDISK_URL) as resp, open(zip_path, "wb") as out:
                out.write(resp.read())
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                raise
            logger.warning("SSL verification failed, retrying without verification")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            with opener.open(IMDISK_URL) as resp, open(zip_path, "wb") as out:
                out.write(resp.read())
        logger.info("Downloaded ImDisk to %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        logger.info("Extracted ImDisk to %s", tmp)
        install_bat = os.path.join(tmp, "install.bat")
        if not os.path.exists(install_bat):
            for root, dirs, files in os.walk(tmp):
                if "install.bat" in files:
                    install_bat = os.path.join(root, "install.bat")
                    break
        if not os.path.exists(install_bat):
            logger.error("install.bat not found in ImDisk archive")
            return False
        logger.info("Running ImDisk installer (UAC elevation required)...")
        ps_cmd = 'Start-Process cmd.exe -ArgumentList "/c cd /d %s && install.bat /fullsilent" -Verb RunAs -Wait' % os.path.dirname(install_bat).replace("'", "''")
        result = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("ImDisk install failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return False
        # install.bat may copy files without registering the driver service; fix up if needed
        sc_query = subprocess.run(["sc", "query", "ImDisk"], capture_output=True, text=True, timeout=10)
        if sc_query.returncode != 0:
            logger.info("Registering ImDisk driver service")
            sc_create = 'Start-Process sc.exe -ArgumentList "create ImDisk type= kernel binPath= system32\\drivers\\imdisk.sys start= auto" -Verb RunAs -Wait'
            subprocess.run(["powershell", "-Command", sc_create], capture_output=True, text=True, timeout=30)
            sc_start = 'Start-Process sc.exe -ArgumentList "start ImDisk" -Verb RunAs -Wait'
            subprocess.run(["powershell", "-Command", sc_start], capture_output=True, text=True, timeout=30)
        logger.info("ImDisk installed successfully")
        return True
    except Exception as e:
        logger.error("Failed to install ImDisk: %s", e)
        return False
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# --- Drive letter helpers ---

def normalize_drive(mount_point: str) -> str:
    """Normalize a Windows drive letter to 'X:' form (no trailing separator)."""
    drive = mount_point.rstrip("\\/")
    if not drive.endswith(":"):
        drive += ":"
    return drive.upper()


def drive_letters():
    """Yield drive letters Z: down to D: as 'X:' strings."""
    for letter in "ZYXWVUTSRQPONMLKJIHGFED":
        yield letter + ":"


def is_our_ramdisk(drive_root: str) -> bool:
    """Check if a drive root has our marker file (from a previous run or crash)."""
    return os.path.exists(os.path.join(drive_root, RAMDISK_MARKER))


def find_existing_ramdisk() -> str:
    """Scan drive letters for a RAM disk left over from a previous crash. Returns 'X:' or empty."""
    for drive in drive_letters():
        drive_root = drive + os.sep
        if os.path.exists(drive_root) and is_our_ramdisk(drive_root):
            logger.info("Found existing RAM disk from previous run: %s", drive)
            return drive
    return ""


def find_free_drive() -> str:
    """Find an unused drive letter, searching from Z: down. Returns 'X:' or empty."""
    for drive in drive_letters():
        if not os.path.exists(drive + os.sep):
            return drive
    return ""


def write_marker(mount: str):
    """Write a marker file to identify our RAM disk after a crash. Raises on failure."""
    marker = os.path.join(mount, RAMDISK_MARKER)
    with open(marker, "w") as f:
        f.write("pid=%d\n" % os.getpid())


# --- Size estimation ---

def estimate_ramdisk_mb(engine: str, reference: str = "", concurrency: int = 1, decompression: int = 2) -> int:
    """Estimate RAM disk size in MB from engine binary sizes and concurrency.

    Formula: (ref_size * conc + eng_size * 2 * conc) * decompression + 128 MB headroom.
    Minimum 256 MB.
    """
    ref_size = 0
    eng_size = 0
    if reference:
        try:
            ref_size = os.path.getsize(reference)
        except OSError as e:
            logger.warning("Cannot stat reference engine %s: %s", reference, e)
    try:
        eng_size = os.path.getsize(engine)
    except OSError as e:
        logger.warning("Cannot stat engine %s: %s", engine, e)
    conc = max(1, concurrency)
    raw = (ref_size * conc + eng_size * 2 * conc) * decompression
    mb = max(256, int(raw / (1024 * 1024)) + 128)
    return mb


# --- Create / remove ---

def create_ramdisk(size_mb: int, drive: str = "", auto_install: bool = True) -> str:
    """Create and format an ImDisk RAM disk on Windows. Returns drive root path.

    If drive is empty, auto-selects an unused drive letter.
    Raises RuntimeError on failure.
    """
    if sys.platform != "win32":
        raise RuntimeError("ImDisk RAM disks are Windows-only; use /dev/shm on Linux")
    # Reuse existing ramdisk if found
    existing = find_existing_ramdisk()
    if existing:
        drive_root = existing + os.sep
        write_marker(drive_root)
        logger.info("Reusing existing RAM disk: %s", existing)
        return drive_root
    # Resolve drive letter
    if drive:
        drive = normalize_drive(drive)
        drive_root = drive + os.sep
        if os.path.exists(drive_root):
            if is_our_ramdisk(drive_root):
                write_marker(drive_root)
                logger.info("Reusing existing RAM disk: %s", drive)
                return drive_root
            raise RuntimeError("Drive %s already exists and is not our RAM disk" % drive)
    else:
        drive = find_free_drive()
        if not drive:
            raise RuntimeError("No free drive letters available for RAM disk")
    # Ensure imdisk is available
    if not has_imdisk():
        if not auto_install:
            raise RuntimeError("ImDisk not found and auto-install is disabled")
        logger.info("ImDisk not found, attempting auto-install")
        if not install_imdisk():
            raise RuntimeError("ImDisk auto-install failed")
        if not has_imdisk():
            raise RuntimeError("ImDisk still not on PATH after install")
    imdisk_cmd = ["imdisk", "-a", "-s", "%dM" % size_mb, "-m", drive]
    logger.info("Creating RAM disk: %s (%d MB)", drive, size_mb)
    result = subprocess.run(imdisk_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0 and auto_install:
        err = (result.stderr or result.stdout or "").strip().lower()
        if "not installed" in err or "does not exist" in err:
            logger.warning("ImDisk driver not available, attempting reinstall")
            if install_imdisk():
                result = subprocess.run(imdisk_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError("imdisk failed (rc=%d): %s" % (result.returncode, (result.stderr or result.stdout or "").strip()))
    # Format requires elevation
    fmt_cmd = 'Start-Process cmd.exe -ArgumentList "/c format %s /fs:ntfs /q /y" -Verb RunAs -Wait' % drive
    result = subprocess.run(["powershell", "-Command", fmt_cmd],
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        subprocess.run(["imdisk", "-D", "-m", drive], capture_output=True, text=True, timeout=10)
        raise RuntimeError("format failed (rc=%d): %s" % (result.returncode, detail))
    drive_root = drive + os.sep
    write_marker(drive_root)
    logger.info("RAM disk created: %s (%d MB)", drive_root, size_mb)
    return drive_root


def remove_ramdisk(drive: str = "") -> bool:
    """Remove an ImDisk RAM disk. Auto-finds our disk if drive is empty. Returns True on success."""
    if sys.platform != "win32":
        raise RuntimeError("ImDisk RAM disks are Windows-only")
    if not drive:
        drive = find_existing_ramdisk()
        if not drive:
            logger.info("No SPSA RAM disk found")
            return False
    drive = normalize_drive(drive)
    logger.info("Removing RAM disk %s", drive)
    result = subprocess.run(["imdisk", "-D", "-m", drive],
                            capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        logger.error("Failed to remove %s (rc=%d): %s", drive, result.returncode, detail)
        return False
    logger.info("RAM disk removed: %s", drive)
    return True


def clean_ramdisk(drive: str = "") -> int:
    """Remove orphaned PyInstaller _MEI* dirs from a RAM disk. Returns count of dirs removed."""
    if not drive:
        drive = find_existing_ramdisk()
        if not drive:
            logger.info("No SPSA RAM disk found")
            return 0
    drive_root = normalize_drive(drive) + os.sep
    if not os.path.isdir(drive_root):
        logger.error("Drive %s does not exist", drive_root)
        return 0
    count = 0
    for entry in os.scandir(drive_root):
        if entry.is_dir() and entry.name.startswith("_MEI"):
            shutil.rmtree(entry.path, ignore_errors=True)
            if not os.path.exists(entry.path):
                count += 1
                logger.info("Removed: %s", entry.path)
            else:
                logger.info("Skipped (in use): %s", entry.path)
    return count


def ramdisk_status() -> list:
    """Scan for SPSA RAM disks. Returns list of (drive, marker_contents) tuples."""
    if sys.platform != "win32":
        shm = "/dev/shm/spsa_temp"
        if os.path.isdir(shm):
            return [(shm, "linux /dev/shm")]
        return []
    found = []
    for drive in drive_letters():
        drive_root = drive + os.sep
        if os.path.exists(drive_root) and is_our_ramdisk(drive_root):
            marker = os.path.join(drive_root, RAMDISK_MARKER)
            try:
                contents = open(marker).read().strip()
            except OSError:
                contents = "(unreadable)"
            # Count _MEI dirs
            mei_count = sum(1 for e in os.scandir(drive_root) if e.is_dir() and e.name.startswith("_MEI"))
            info = contents
            if mei_count:
                info += ", %d _MEI dir(s)" % mei_count
            found.append((drive, info))
    return found


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="SPSA RAM disk management tool")
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a RAM disk for engine temp files")
    p_create.add_argument("--engine", required=True, help="Path to engine binary")
    p_create.add_argument("--ref", default="", help="Path to reference engine binary")
    p_create.add_argument("--concurrency", type=int, default=os.cpu_count() or 1,
                          help="Total concurrent engine processes (default: CPU count)")
    p_create.add_argument("--drive", default="", help="Drive letter (e.g., Z:); auto-select if omitted")
    p_create.add_argument("--size", type=int, default=0, help="Override size in MB (0 = auto-estimate)")
    p_create.add_argument("--factor", type=int, default=2,
                          help="Decompression multiplier for size estimation (default: 2)")
    p_create.add_argument("--no-install", action="store_true", help="Don't auto-install ImDisk if missing")

    # remove
    p_remove = sub.add_parser("remove", help="Remove an SPSA RAM disk")
    p_remove.add_argument("--drive", default="", help="Drive letter to remove; auto-detect if omitted")

    # clean
    p_clean = sub.add_parser("clean", help="Remove orphaned _MEI* dirs from RAM disk")
    p_clean.add_argument("--drive", default="", help="Drive letter to clean; auto-detect if omitted")

    # status
    sub.add_parser("status", help="Show active SPSA RAM disks")

    args = parser.parse_args()

    # Setup logging to console
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    if args.command == "create":
        if sys.platform != "win32":
            print("Linux: use /dev/shm (already RAM-backed, no setup needed)")
            sys.exit(0)
        if not os.path.isfile(args.engine):
            print("Error: engine not found: %s" % args.engine, file=sys.stderr)
            sys.exit(1)
        if args.ref and not os.path.isfile(args.ref):
            print("Error: reference engine not found: %s" % args.ref, file=sys.stderr)
            sys.exit(1)
        size_mb = args.size
        if size_mb <= 0:
            size_mb = estimate_ramdisk_mb(args.engine, args.ref, args.concurrency, args.factor)
        print("Engine: %s (%d MB)" % (args.engine, os.path.getsize(args.engine) // (1024 * 1024)))
        if args.ref:
            print("Reference: %s (%d MB)" % (args.ref, os.path.getsize(args.ref) // (1024 * 1024)))
        print("Concurrency: %d, decompression factor: %dx" % (args.concurrency, args.factor))
        print("Estimated size: %d MB" % size_mb)
        try:
            drive_root = create_ramdisk(size_mb, args.drive, auto_install=not args.no_install)
            print("RAM disk ready: %s" % drive_root)
        except RuntimeError as e:
            print("Error: %s" % e, file=sys.stderr)
            sys.exit(1)

    elif args.command == "remove":
        if sys.platform != "win32":
            shm = "/dev/shm/spsa_temp"
            if os.path.isdir(shm):
                shutil.rmtree(shm, ignore_errors=True)
                print("Removed: %s" % shm)
            else:
                print("No SPSA temp dir found")
            sys.exit(0)
        if not remove_ramdisk(args.drive):
            sys.exit(1)

    elif args.command == "clean":
        if sys.platform != "win32":
            print("Clean is Windows-only (_MEI dirs are a PyInstaller/Windows issue)")
            sys.exit(0)
        count = clean_ramdisk(args.drive)
        print("Cleaned %d orphaned _MEI dir(s)" % count)

    elif args.command == "status":
        found = ramdisk_status()
        if not found:
            print("No SPSA RAM disks found")
        else:
            for drive, info in found:
                print("  %s  %s" % (drive, info))


if __name__ == "__main__":
    main()
