from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import os
import platform
import subprocess
import time
from pathlib import Path


def _format_gb(n_bytes: int | float | None) -> str:
    if n_bytes is None:
        return ""
    return f"{float(n_bytes) / (1024 ** 3):.3f}"


def _system_memory_windows() -> tuple[int, int] | None:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    return int(stat.ullTotalPhys), int(stat.ullAvailPhys)


def _system_memory_proc() -> tuple[int, int] | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    vals: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            vals[parts[0].rstrip(":")] = int(parts[1]) * 1024
    total = vals.get("MemTotal")
    available = vals.get("MemAvailable", vals.get("MemFree"))
    if total is None or available is None:
        return None
    return int(total), int(available)


def system_memory() -> tuple[int | None, int | None, int | None, float | None]:
    if platform.system().lower() == "windows":
        pair = _system_memory_windows()
    else:
        pair = _system_memory_proc()
    if pair is None:
        return None, None, None, None
    total, available = pair
    used = total - available
    pct = (used / total * 100.0) if total else None
    return total, available, used, pct


def _process_rss_with_psutil(pid: int | None, name: str | None) -> tuple[int | None, int | None]:
    try:
        import psutil  # type: ignore
    except Exception:
        return None, None

    matches = []
    if pid is not None:
        try:
            proc = psutil.Process(int(pid))
            if proc.is_running():
                matches.append(proc)
        except Exception:
            pass
    elif name:
        needle = str(name).lower()
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                pname = str(proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if needle in pname or needle in cmdline:
                    matches.append(proc)
            except Exception:
                continue

    rss_total = 0
    count = 0
    for proc in matches:
        try:
            rss_total += int(proc.memory_info().rss)
            count += 1
        except Exception:
            continue
    return (rss_total, count) if count else (None, 0)


def _process_rss_windows_tasklist(pid: int | None, name: str | None) -> tuple[int | None, int | None]:
    if platform.system().lower() != "windows":
        return None, None
    cmd = ["tasklist", "/FO", "CSV"]
    if pid is not None:
        cmd.extend(["/FI", f"PID eq {int(pid)}"])
    elif name:
        cmd.extend(["/FI", f"IMAGENAME eq {name}"])
    else:
        return None, None

    try:
        out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    except Exception:
        return None, None

    rss_total = 0
    count = 0
    for row in csv.DictReader(out.splitlines()):
        mem = row.get("Mem Usage") or ""
        digits = "".join(ch for ch in mem if ch.isdigit())
        if not digits:
            continue
        rss_total += int(digits) * 1024
        count += 1
    return (rss_total, count) if count else (None, 0)


def process_rss(pid: int | None, name: str | None) -> tuple[int | None, int | None]:
    rss, count = _process_rss_with_psutil(pid, name)
    if rss is not None or count:
        return rss, count
    return _process_rss_windows_tasklist(pid, name)


def write_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(
            "timestamp\t"
            "system_used_gb\t"
            "system_available_gb\t"
            "system_total_gb\t"
            "system_used_percent\t"
            "process_rss_gb\t"
            "process_count\n"
        )


def log_once(path: Path, pid: int | None, name: str | None) -> None:
    total, available, used, pct = system_memory()
    proc_rss, proc_count = process_rss(pid, name)
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    pct_text = f"{pct:.2f}" if pct is not None else ""
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(
            f"{timestamp}\t"
            f"{_format_gb(used)}\t"
            f"{_format_gb(available)}\t"
            f"{_format_gb(total)}\t"
            f"{pct_text}\t"
            f"{_format_gb(proc_rss)}\t"
            f"{int(proc_count or 0)}\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Log RAM usage to a tab-separated text file.")
    parser.add_argument("--out", default="ram_usage_log.txt", help="Output text file. Default: ram_usage_log.txt")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between samples. Default: 60")
    parser.add_argument("--pid", type=int, default=None, help="Optional process ID to track.")
    parser.add_argument("--name", default=None, help="Optional process image/name to track, e.g. python.exe.")
    args = parser.parse_args()

    if args.pid is not None and args.name:
        parser.error("Use either --pid or --name, not both.")

    out = Path(args.out).expanduser().resolve()
    write_header(out)
    print(f"Logging RAM usage every {float(args.interval):g}s to {out}")
    if args.pid is not None:
        print(f"Tracking process PID {int(args.pid)}")
    elif args.name:
        print(f"Tracking processes matching {args.name!r}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            log_once(out, args.pid, args.name)
            time.sleep(max(1.0, float(args.interval)))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
