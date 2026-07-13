"""Install / monitor mimesys workers over SSH.

Usage:
  uv run python install_remote_dependencies.py              # bootstrap install on every host in config.HOSTNAMES
  uv run python install_remote_dependencies.py --status     # one-shot status snapshot (running / ok / failed)
  uv run python install_remote_dependencies.py --wait       # block until every host reaches a terminal state
  uv run python install_remote_dependencies.py --logs       # print the last lines of install_run.log per host
  uv run python install_remote_dependencies.py --install --wait   # install + wait in one shot

Status reporting relies on install_executor.sh writing to
~/.mimesys_install.status — values are "running stage=<name> ...", "ok ...",
or "failed exit=<code> stage=<name> ...".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from dataclasses import dataclass

import paramiko

# Host config: override via MIMESYS_WORKER_CONFIG_PATH or --config-path
# (defaults to config.py next to this script); loaded under the name `config`.
_CFG_PATH = os.environ.get("MIMESYS_WORKER_CONFIG_PATH")
if "--config-path" in sys.argv:
    _ci = sys.argv.index("--config-path")
    _CFG_PATH = sys.argv[_ci + 1]
    del sys.argv[_ci:_ci + 2]
if _CFG_PATH:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("config", _CFG_PATH)
    config = _ilu.module_from_spec(_spec); _spec.loader.exec_module(config)
    print(f"[config] using {_CFG_PATH}  HOSTS={len(config.HOSTNAMES)}")
else:
    import config


# Path to the per-host status file written by install_executor.sh.
STATUS_FILE = f"{config.REMOTE_HOME_DIR}/.mimesys_install.status"
LOG_FILE = f"{config.REMOTE_HOME_DIR}/install_run.log"
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Connection helpers ────────────────────────────────────────────────────

def _key():
    return paramiko.RSAKey.from_private_key_file(os.path.expanduser(config.PRIVATE_KEY_PATH))


def _connect(hostname: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, port=config.PORT, username=config.USERNAME, pkey=_key())
    return client


def _run(client: paramiko.SSHClient, cmd: str) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    return exit_status, stdout.read().decode(), stderr.read().decode()


# ─── Install ────────────────────────────────────────────────────────────────

def install_host(hostname: str) -> None:
    """Push scripts + SSH keys, clone HPCPerfStats, kick off install_executor.sh in tmux."""
    print(f"[{hostname}] starting install")
    client = _connect(hostname)
    try:
        scp = paramiko.SFTPClient.from_transport(client.get_transport())

        # SSH keys + ssh config so the worker can:
        #   * clone github repos during install (typically id_rsa works for github),
        #   * scp results back to the controller (which usually means matching the
        #     IdentityFile referenced by Host MY_HOSTNAME in ~/.ssh/config — for
        #     this user that's id_rsa_utns via a ProxyCommand through dex).
        # We always copy id_rsa (used by install_executor.sh's git clone) AND the
        # PRIVATE_KEY_PATH key (used by collect_mimesys_metrics.sh's scp-back).
        scp.put(f"{config.LOCAL_HOME_DIR}/.ssh/config", f"{config.REMOTE_HOME_DIR}/.ssh/config")
        ssh_files = ["id_rsa", "id_rsa.pub"]
        priv_basename = os.path.basename(os.path.expanduser(config.PRIVATE_KEY_PATH))
        if priv_basename not in ssh_files:
            ssh_files.append(priv_basename)
            pub = priv_basename + ".pub"
            if os.path.exists(os.path.expanduser(f"{config.LOCAL_HOME_DIR}/.ssh/{pub}")):
                ssh_files.append(pub)
        for fname in ssh_files:
            local = os.path.expanduser(f"{config.LOCAL_HOME_DIR}/.ssh/{fname}")
            if not os.path.exists(local):
                print(f"[{hostname}] warning: {local} not found locally, skipping")
                continue
            scp.put(local, f"{config.REMOTE_HOME_DIR}/.ssh/{fname}")
        _run(client, (
            f"chmod 600 {config.REMOTE_HOME_DIR}/.ssh/id_rsa "
            f"{config.REMOTE_HOME_DIR}/.ssh/{priv_basename} 2>/dev/null || true"
        ))

        for script in ("install_executor.sh", "stats.x", "collect_mimesys_metrics.sh"):
            scp.put(f"{SCRIPTS_DIR}/{script}", f"{config.REMOTE_HOME_DIR}/{script}")

        # Reset the status sentinel so a re-run doesn't show stale "ok"/"failed"
        # before install_executor.sh's first mark_stage runs.
        _run(client, f"echo 'pending ts=$(date +%s)' > {STATUS_FILE}")

        # HPCPerfStats source (idempotent: skip clone if already present).
        _run(client, (
            f"test -d {config.REMOTE_HOME_DIR}/HPCPerfStats || "
            f"(cd {config.REMOTE_HOME_DIR} && git clone https://github.com/TACC/HPCPerfStats.git "
            f"&& cd HPCPerfStats && git checkout 2.4)"
        ))

        # Kill any prior tmux session before relaunch so re-runs are clean.
        _run(client, "tmux kill-session -t install 2>/dev/null || true")
        ec, out, err = _run(client, (
            f"tmux new-session -d -s install "
            f"'bash {config.REMOTE_HOME_DIR}/install_executor.sh > {LOG_FILE} 2>&1'"
        ))
        if ec != 0:
            print(f"[{hostname}] tmux launch failed (exit={ec}): {err.strip()}")
        else:
            print(f"[{hostname}] install running in tmux session 'install' "
                  f"(check status: --status, watch: --wait)")
    except Exception as e:
        print(f"[{hostname}] error: {e}")
    finally:
        client.close()


# ─── Status / wait ─────────────────────────────────────────────────────────

@dataclass
class HostStatus:
    hostname: str
    state: str          # one of: running / ok / failed / pending / missing / unreachable
    raw: str            # raw status-file line
    tmux_alive: bool    # whether the install tmux session still exists


def query_status(hostname: str) -> HostStatus:
    try:
        client = _connect(hostname)
    except Exception as e:
        return HostStatus(hostname, "unreachable", str(e), False)
    try:
        ec, out, _ = _run(client, f"cat {STATUS_FILE} 2>/dev/null")
        raw = out.strip()
        if ec != 0 or not raw:
            state = "missing"
        else:
            state = raw.split()[0]   # running / ok / failed / pending
        ec2, out2, _ = _run(client, "tmux has-session -t install 2>/dev/null && echo 1 || echo 0")
        tmux_alive = out2.strip() == "1"
        return HostStatus(hostname, state, raw, tmux_alive)
    finally:
        client.close()


def fetch_log_tail(hostname: str, n: int = 25) -> str:
    try:
        client = _connect(hostname)
    except Exception as e:
        return f"unreachable: {e}"
    try:
        _, out, _ = _run(client, f"tail -n {n} {LOG_FILE} 2>/dev/null")
        return out.rstrip()
    finally:
        client.close()


def _format_status(s: HostStatus) -> str:
    icon = {
        "ok": "✓",
        "failed": "✗",
        "running": "…",
        "pending": "·",
        "missing": "?",
        "unreachable": "!",
    }.get(s.state, "?")
    tmux_note = "" if s.state in ("ok", "failed") else f"  tmux={'alive' if s.tmux_alive else 'gone'}"
    return f"{icon} {s.hostname:50s} {s.raw or '(no status)'}{tmux_note}"


def cmd_status(hostnames: list[str]) -> int:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(hostnames))) as ex:
        results = list(ex.map(query_status, hostnames))
    for s in results:
        print(_format_status(s))
    bad = [s for s in results if s.state in ("failed", "unreachable")]
    return 1 if bad else 0


def cmd_logs(hostnames: list[str], n: int) -> int:
    for h in hostnames:
        print(f"\n========== {h}  (last {n} lines of install_run.log) ==========")
        print(fetch_log_tail(h, n=n))
    return 0


def cmd_wait(hostnames: list[str], poll_seconds: int = 30, timeout_seconds: int = 3600) -> int:
    """Poll until every host reaches a terminal state (ok / failed / unreachable)."""
    start = time.monotonic()
    last_states: dict[str, str] = {}
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(hostnames))) as ex:
            results = list(ex.map(query_status, hostnames))
        # Print only on state changes, plus a heartbeat with the host counts.
        changed = False
        for s in results:
            if last_states.get(s.hostname) != s.raw:
                print(_format_status(s))
                last_states[s.hostname] = s.raw
                changed = True
        if not changed:
            counts = {}
            for s in results:
                counts[s.state] = counts.get(s.state, 0) + 1
            elapsed = int(time.monotonic() - start)
            print(f"  [{elapsed:5d}s] " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if all(s.state in ("ok", "failed", "unreachable") for s in results):
            print("\n=== final ===")
            for s in results:
                print(_format_status(s))
            return 0 if all(s.state == "ok" for s in results) else 1
        if time.monotonic() - start > timeout_seconds:
            print(f"  timeout after {timeout_seconds}s; not all hosts reached a terminal state")
            return 2
        time.sleep(poll_seconds)


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true",
                    help="Bootstrap install on every host in config.HOSTNAMES (default if no other action is given)")
    ap.add_argument("--status", action="store_true",
                    help="Print one-shot status for every host and exit")
    ap.add_argument("--wait", action="store_true",
                    help="Poll until every host reaches a terminal state (ok/failed)")
    ap.add_argument("--logs", action="store_true",
                    help="Print the tail of install_run.log per host")
    ap.add_argument("--tail-lines", type=int, default=25,
                    help="Number of log lines to show with --logs (default 25)")
    ap.add_argument("--poll-seconds", type=int, default=30, help="Wait poll cadence")
    ap.add_argument("--timeout-seconds", type=int, default=3600, help="Wait timeout")
    args = ap.parse_args()

    # Default behaviour (no flags) is install-only, matching the original script.
    if not (args.install or args.status or args.wait or args.logs):
        args.install = True

    if not config.HOSTNAMES:
        print("config.HOSTNAMES is empty — nothing to do.", file=sys.stderr)
        return 1

    rc = 0
    if args.install:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            list(ex.map(install_host, config.HOSTNAMES))
    if args.status:
        rc = cmd_status(config.HOSTNAMES) or rc
    if args.logs:
        rc = cmd_logs(config.HOSTNAMES, args.tail_lines) or rc
    if args.wait:
        rc = cmd_wait(config.HOSTNAMES, args.poll_seconds, args.timeout_seconds) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
