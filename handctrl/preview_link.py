"""Daemon side of the pop-out preview: spawns the Qt window process and streams to it."""

from __future__ import annotations

import json
import shutil
import socket
import struct
import subprocess
import threading
from pathlib import Path

import cv2

SCRIPT = Path(__file__).with_name("preview.py")


class PreviewLink:
    def __init__(self, python: str = "/usr/bin/python3"):
        self.python = python if shutil.which(python) else "python3"
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self._slot = None
        self._cond = threading.Condition()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.alive:
            return
        ours, theirs = socket.socketpair()
        self.proc = subprocess.Popen(
            [self.python, str(SCRIPT), str(theirs.fileno())],
            pass_fds=[theirs.fileno()], stdin=subprocess.DEVNULL,
        )
        theirs.close()
        self.sock = ours
        threading.Thread(target=self._send_loop, args=(ours,), name="preview", daemon=True).start()

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        with self._cond:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
            self._cond.notify_all()

    def toggle(self) -> bool:
        if self.alive:
            self.stop()
        else:
            self.start()
        return self.alive

    def push(self, frame, info: dict) -> None:
        """Offer the newest frame; dropped if the sender is still busy (never blocks)."""
        if not self.alive:
            return
        with self._cond:
            self._slot = (frame, info)
            self._cond.notify()

    def _send_loop(self, sock: socket.socket) -> None:
        while True:
            with self._cond:
                while self._slot is None and self.sock is sock:
                    self._cond.wait(1.0)
                    if not self.alive:
                        return
                if self.sock is not sock:
                    return
                frame, info = self._slot
                self._slot = None
            h, w = frame.shape[:2]
            if w > 320:
                frame = cv2.resize(frame, (320, round(h * 320 / w)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            head = json.dumps(info, separators=(",", ":")).encode()
            try:
                sock.sendall(struct.pack("<I", len(head)) + head
                             + struct.pack("<I", len(jpg)) + jpg.tobytes())
            except OSError:
                return
