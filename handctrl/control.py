"""Tiny line-based control socket so the CLI / a KDE shortcut can drive the daemon."""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

SOCK_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "handctrl.sock"


def send(cmd: str, timeout: float = 3.0) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(SOCK_PATH))
        s.sendall(cmd.encode() + b"\n")
        return s.recv(4096).decode().strip()


def is_running() -> bool:
    try:
        send("status", timeout=1.0)
        return True
    except OSError:
        return False


class ControlServer:
    def __init__(self, handler):
        self.handler = handler
        if SOCK_PATH.exists():
            SOCK_PATH.unlink()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(SOCK_PATH))
        os.chmod(SOCK_PATH, 0o600)
        self.sock.listen(4)
        threading.Thread(target=self._serve, name="control", daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(2.0)
                    cmd = conn.recv(256).decode().strip()
                    conn.sendall((self.handler(cmd) + "\n").encode())
                except OSError:
                    pass

    def close(self) -> None:
        self.sock.close()
        try:
            SOCK_PATH.unlink()
        except FileNotFoundError:
            pass
