"""Pop-out live camera preview for handctrl.

Runs as its own process (system Python + PySide6) so the daemon never loads Qt and pays
nothing while the preview is closed. Receives frames over an inherited socket:
    [u32 header_len][json header][u32 jpeg_len][jpeg bytes]
Controls: drag to move · scroll to resize · double-click or Esc to close.
"""

import json
import os
import socket
import struct
import sys
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # XWayland: lets KWin honour keep-above

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QWidget

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
         (17, 18), (18, 19), (19, 20)]
TIPS = {"index": 8, "middle": 12}
ACCENT = {"index": QColor(64, 156, 255), "middle": QColor(175, 110, 255),
          "micro": QColor(48, 200, 180)}
LABEL = {
    ("index", "press"): "Click", ("index", "drag"): "Dragging",
    ("middle", "pending"): "Right click · hold to scroll", ("middle", "scroll"): "Scrolling",
}
HELP = ("Move: open hand\nIndex pinch: click / hold to drag\n"
        "Middle pinch: right click / hold+move to scroll\n"
        "Index relaxed, other fingers curled, thumb on index side:\n  tap = middle click,"
        "  swipe ⇆ = desktops, swipe ↑ = overview\n"
        "Hold fist: pause / resume\n\n"
        "Drag to move · scroll to resize · H toggles help · double-click to close")
BASE_W, BASE_H = 320, 240
PANEL_W = 210
ROWS = [  # key, title, subtitle, colour
    ("move", "Open hand", "Move the cursor", QColor(235, 235, 235)),
    ("index", "Index + thumb", "Click · hold to drag", ACCENT["index"]),
    ("middle", "Middle + thumb", "Right click · hold to scroll", ACCENT["middle"]),
    ("micro", "Thumb on index side", "Tap: middle click · swipe ⇆ ↑", ACCENT["micro"]),
    ("fist", "Closed fist ~1 s", "Pause / resume", QColor(255, 204, 0)),
]


class Feed(QObject):
    frame = Signal(dict, bytes)
    closed = Signal()

    def __init__(self, sock):
        super().__init__()
        self.sock = sock
        threading.Thread(target=self._run, daemon=True).start()

    def _read(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return bytes(buf)

    def _run(self):
        try:
            while True:
                (hl,) = struct.unpack("<I", self._read(4))
                head = json.loads(self._read(hl))
                (jl,) = struct.unpack("<I", self._read(4))
                self.frame.emit(head, self._read(jl))
        except (EOFError, OSError, ValueError):
            self.closed.emit()


class Preview(QWidget):
    def __init__(self, feed):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("Hand Control preview")
        self.setToolTip(HELP)
        self.scale = 1.0
        self.panel = True
        self.img = None
        self.info = {}
        self._drag = None
        self._resize()
        scr = QGuiApplication.primaryScreen().availableGeometry()
        self.move(scr.right() - self.width() - 24, scr.top() + 24)
        feed.frame.connect(self.on_frame)
        feed.closed.connect(QApplication.quit)

    def on_frame(self, info, jpeg):
        img = QImage.fromData(jpeg, "JPG")
        if not img.isNull():
            self.img = img.flipped(Qt.Horizontal) if hasattr(img, "flipped") else img.mirrored(True, False)
        self.info = info
        self.update()

    # ---- painting -------------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        full = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(full, 16, 16)
        p.setClipPath(clip)
        p.fillRect(full, QColor(18, 18, 22))
        r = QRectF(0, 0, BASE_W * self.scale, full.height())
        info = self.info
        paused = info.get("state") != "active"
        if self.img is not None:
            p.setOpacity(0.55 if paused else 0.9)
            p.drawImage(r, self.img)
            p.setOpacity(1.0)
        W, H = r.width(), r.height()

        def pt(x, y):  # landmark (unmirrored normalized) -> widget point
            return QPointF((1.0 - x) * W, y * H)

        # pointer area: dim outside, outline inside, cursor dot = where the pointer is
        box = info.get("box")
        if box:
            br = QRectF(box[0] * W, box[1] * H, (box[2] - box[0]) * W, (box[3] - box[1]) * H)
            outside = QPainterPath()
            outside.addRect(r)
            inner = QPainterPath()
            inner.addRoundedRect(br, 10, 10)
            p.fillPath(outside.subtracted(inner), QColor(0, 0, 0, 110))
            p.setPen(QPen(QColor(255, 255, 255, 90), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(br, 10, 10)
            cur = info.get("cursor")
            if cur:
                c = QPointF(br.left() + cur[0] * br.width(), br.top() + cur[1] * br.height())
                p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
                p.drawLine(c - QPointF(6, 0), c + QPointF(6, 0))
                p.drawLine(c - QPointF(0, 6), c + QPointF(0, 6))

        lm = info.get("lm")
        finger = info.get("finger")
        if lm:
            pts = [pt(x, y) for x, y in lm]
            p.setPen(QPen(QColor(255, 255, 255, 150 if not paused else 70), 2))
            for a, b in BONES:
                p.drawLine(pts[a], pts[b])
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 200))
            for q in pts:
                p.drawEllipse(q, 2.2, 2.2)
            self._pinch_rings(p, pts, info, finger, paused)
            anchor = info.get("anchor")
            if anchor:
                a = QPointF(anchor[0] * W, anchor[1] * H)
                p.setBrush(QColor(255, 255, 255, 230))
                p.drawEllipse(a, 4, 4)

        self._hud(p, r, info, finger, paused)
        if self.panel:
            self._help(p, QRectF(r.right(), 0, full.width() - r.width(), full.height()),
                       info, finger, paused)
        p.end()

    def _active_row(self, info, finger, paused):
        if info.get("fist", 0) > 0:
            return "fist"
        if paused or not info.get("lm"):
            return None
        if info.get("mode") == "micro":
            return "micro"
        return finger or "move"

    def _help(self, p, rect, info, finger, paused):
        """Gesture cheat-sheet beside the camera; the gesture in use lights up."""
        k = self.scale
        p.fillRect(rect, QColor(24, 24, 30))
        p.setPen(QPen(QColor(255, 255, 255, 25), 1))
        p.drawLine(rect.topLeft(), rect.bottomLeft())
        active = self._active_row(info, finger, paused)
        title, sub = QFont(), QFont()
        title.setPointSizeF(8.5 * k)
        title.setBold(True)
        sub.setPointSizeF(7.5 * k)
        x, y = rect.left() + 10 * k, rect.top() + 10 * k
        p.setFont(title)
        p.setPen(QColor(255, 255, 255, 140))
        p.drawText(QPointF(x, y + 10 * k), "GESTURES")
        y += 20 * k
        row_h = (rect.bottom() - y - 18 * k) / len(ROWS)
        for key, name, desc, col in ROWS:
            row = QRectF(rect.left() + 5 * k, y, rect.width() - 10 * k, row_h - 3 * k)
            on = key == active
            if on:
                bg = QColor(col)
                bg.setAlpha(55)
                p.setPen(Qt.NoPen)
                p.setBrush(bg)
                p.drawRoundedRect(row, 8 * k, 8 * k)
            dot = QColor(col)
            dot.setAlpha(255 if on else 150)
            p.setPen(Qt.NoPen)
            p.setBrush(dot)
            p.drawEllipse(QPointF(x + 4 * k, row.center().y()), 4 * k, 4 * k)
            tx = x + 15 * k
            p.setFont(title)
            p.setPen(QColor(255, 255, 255, 255 if on else 200))
            p.drawText(QRectF(tx, row.top(), row.right() - tx, row.height() / 2 + 2 * k),
                       Qt.AlignLeft | Qt.AlignBottom, name)
            p.setFont(sub)
            p.setPen(QColor(255, 255, 255, 200 if on else 120))
            p.drawText(QRectF(tx, row.center().y() + 1 * k, row.right() - tx, row.height() / 2),
                       Qt.AlignLeft | Qt.AlignTop, desc)
            y += row_h
        p.setFont(sub)
        p.setPen(QColor(255, 255, 255, 90))
        p.drawText(QRectF(rect.left(), rect.bottom() - 16 * k, rect.width(), 14 * k),
                   Qt.AlignCenter, "H hide help · scroll resize · dbl-click close")

    def _pinch_rings(self, p, pts, info, finger, paused):
        """Vision-Pro style: a ring between thumb and each finger shrinks as they close,
        and fills with the gesture colour once the pinch engages."""
        dist = info.get("dist") or {}
        on, off = info.get("pinch_on", 0.22), info.get("pinch_off", 0.35)
        thumb = pts[4]
        for name, tip in TIPS.items():
            d = dist.get(name)
            if d is None or (d > off * 2.2 and name != finger):
                continue
            col = QColor(ACCENT[name])
            mid = (thumb + pts[tip]) / 2
            closeness = max(0.0, min(1.0, (off * 2.2 - d) / (off * 2.2 - on)))
            radius = 4 + 10 * (1 - closeness)
            if name == finger and not paused:
                col.setAlpha(230)
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawEllipse(mid, 9, 9)
            else:
                col.setAlpha(int(60 + 170 * closeness))
                p.setPen(QPen(col, 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(mid, radius, radius)

    def _pill(self, p, x, y, text, color, anchor_bottom=False, right=False):
        f = QFont()
        f.setPointSizeF(8.5 * max(0.8, self.scale))
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 18, fm.height() + 8
        if right:
            x -= w
        if anchor_bottom:
            y -= h
        rr = QRectF(x, y, w, h)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 15, 20, 190))
        p.drawRoundedRect(rr, h / 2, h / 2)
        p.setPen(color)
        p.drawText(rr, Qt.AlignCenter, text)

    def _hud(self, p, r, info, finger, paused):
        state = info.get("state", "…")
        dot = {"active": QColor(52, 199, 89), "paused": QColor(255, 204, 0)}.get(
            state, QColor(160, 160, 160))
        self._pill(p, 8, 8, f"● {state.upper()}", dot)
        self._pill(p, r.width() - 8, 8,
                   f"{info.get('delegate', '').upper()} · {info.get('fps', 0):.0f} fps",
                   QColor(220, 220, 220), right=True)
        fist = info.get("fist", 0.0)
        if info.get("calib"):
            text, col = f"Calibrating {info['calib']:.0f}s — sweep your hand", QColor(255, 204, 0)
        elif fist > 0:
            verb = "resume" if paused else "pause"
            text, col = f"Hold fist to {verb}… {int(fist * 100)}%", QColor(255, 204, 0)
        elif paused:
            text, col = "Paused — hold a fist to resume", QColor(200, 200, 200)
        elif not info.get("lm"):
            text, col = "Show your hand", QColor(200, 200, 200)
        elif finger:
            text = LABEL.get((finger, info.get("mode")), finger.title() + " pinch")
            col = ACCENT[finger].lighter(130)
        elif info.get("mode") == "micro":
            text, col = "Thumb: tap · swipe ⇆ desktops · ↑ overview", ACCENT["micro"].lighter(130)
        else:
            text, col = "Pointing", QColor(235, 235, 235)
        self._pill(p, 8, r.height() - 8, text, col, anchor_bottom=True)
        if fist > 0:  # progress arc
            c = QPointF(r.width() - 22, r.height() - 22)
            p.setPen(QPen(QColor(255, 255, 255, 60), 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 11, 11)
            p.setPen(QPen(QColor(255, 204, 0), 3, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(QRectF(c.x() - 11, c.y() - 11, 22, 22), 90 * 16, int(-360 * 16 * fist))

    # ---- interaction ----------------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                return
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        QApplication.quit()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            QApplication.quit()
        elif e.key() == Qt.Key_H:
            self.panel = not self.panel
            self._resize()

    def wheelEvent(self, e):
        step = 1.1 if e.angleDelta().y() > 0 else 1 / 1.1
        self.scale = max(0.6, min(2.5, self.scale * step))
        self._resize()

    def _resize(self):
        w = BASE_W + (PANEL_W if self.panel else 0)
        self.resize(int(w * self.scale), int(BASE_H * self.scale))


def main():
    sock = socket.socket(fileno=int(sys.argv[1]))
    app = QApplication(sys.argv[:1])
    app.setApplicationName("handctrl-preview")
    feed = Feed(sock)
    w = Preview(feed)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
