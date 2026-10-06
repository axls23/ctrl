"""System tray icon via the StatusNotifierItem D-Bus protocol (KDE native; GNOME with the
AppIndicator extension). Talks D-Bus directly with dbus-fast — no Qt/GTK in the process."""

import asyncio
import os
import threading

from dbus_fast import PropertyAccess, Variant
from dbus_fast.aio import MessageBus
from dbus_fast.service import ServiceInterface, dbus_property, method, signal

ICONS = {"active": "transform-browse", "paused": "media-playback-pause", "off": "camera-off"}
LABELS = {"active": "Active", "paused": "Paused", "off": "Off (camera released)"}
HELP = (
    "Move: relaxed open hand\n"
    "Index+thumb pinch: click / hold to drag\n"
    "Middle+thumb pinch: right click / hold+move to scroll\n"
    "Index relaxed, others curled, thumb on index side: tap = middle click,\n"
    "  swipe left/right = switch desktop, swipe up = overview\n"
    "Closed fist (hold ~1s): pause / resume"
)


class _Item(ServiceInterface):
    def __init__(self, tray):
        super().__init__("org.kde.StatusNotifierItem")
        self.tray = tray

    @dbus_property(access=PropertyAccess.READ)
    def Category(self) -> "s":
        return "Hardware"

    @dbus_property(access=PropertyAccess.READ)
    def Id(self) -> "s":
        return "handctrl"

    @dbus_property(access=PropertyAccess.READ)
    def Title(self) -> "s":
        return "Hand Control"

    @dbus_property(access=PropertyAccess.READ)
    def Status(self) -> "s":
        return "Active"  # "Passive" would hide the icon in the overflow area

    @dbus_property(access=PropertyAccess.READ)
    def IconName(self) -> "s":
        return ICONS[self.tray.state]

    @dbus_property(access=PropertyAccess.READ)
    def ToolTip(self) -> "(sa(iiay)ss)":
        return [ICONS[self.tray.state], [], "Hand Control: " + LABELS[self.tray.state],
                "Click to toggle. Closed fist also pauses/resumes."]

    @dbus_property(access=PropertyAccess.READ)
    def ItemIsMenu(self) -> "b":
        return False

    @dbus_property(access=PropertyAccess.READ)
    def Menu(self) -> "o":
        return "/MenuBar"

    @method()
    def Activate(self, x: "i", y: "i"):
        self.tray.command("toggle")

    @method()
    def SecondaryActivate(self, x: "i", y: "i"):
        self.tray.command("off" if self.tray.state != "off" else "active")

    @method()
    def ContextMenu(self, x: "i", y: "i"):
        pass

    @method()
    def Scroll(self, delta: "i", orientation: "s"):
        pass

    @signal()
    def NewIcon(self):
        pass

    @signal()
    def NewToolTip(self):
        pass

    @signal()
    def NewStatus(self, status: "s") -> "s":
        return status


class _Menu(ServiceInterface):
    # id -> (label, command or None, kind)
    ITEMS = {
        1: ("Active", "active", "radio"),
        2: ("Paused", "paused", "radio"),
        3: (LABELS["off"], "off", "radio"),
        4: ("", None, "separator"),
        5: ("Show / hide camera preview", "preview", ""),
        9: ("Calibrate pointer area…", "calibrate", ""),
        6: ("Gesture cheat sheet", "help", ""),
        7: ("", None, "separator"),
        8: ("Quit", "quit", ""),
    }

    def __init__(self, tray):
        super().__init__("com.canonical.dbusmenu")
        self.tray = tray
        self.revision = 1

    def _props(self, i):
        label, cmd, kind = self.ITEMS[i]
        if kind == "separator":
            return {"type": Variant("s", "separator")}
        p = {"label": Variant("s", label)}
        if kind == "radio":
            p["toggle-type"] = Variant("s", "radio")
            p["toggle-state"] = Variant("i", int(self.tray.state == cmd))
        return p

    @method()
    def GetLayout(self, parent: "i", depth: "i", names: "as") -> "u(ia{sv}av)":
        children = [Variant("(ia{sv}av)", [i, self._props(i), []]) for i in self.ITEMS]
        if parent != 0:
            return [self.revision, [parent, self._props(parent), []]]
        return [self.revision, [0, {"children-display": Variant("s", "submenu")}, children]]

    @method()
    def GetGroupProperties(self, ids: "ai", names: "as") -> "a(ia{sv})":
        return [[i, self._props(i)] for i in (ids or self.ITEMS) if i in self.ITEMS]

    @method()
    def GetProperty(self, id: "i", name: "s") -> "v":
        return self._props(id).get(name, Variant("s", ""))

    @method()
    def Event(self, id: "i", event: "s", data: "v", timestamp: "u"):
        self._event(id, event)

    @method()
    def EventGroup(self, events: "a(isvu)") -> "ai":
        for id, event, _data, _ts in events:
            self._event(id, event)
        return []

    def _event(self, id, event):
        if event == "clicked" and id in self.ITEMS and self.ITEMS[id][1]:
            self.tray.command(self.ITEMS[id][1])

    @method()
    def AboutToShow(self, id: "i") -> "b":
        return False

    @method()
    def AboutToShowGroup(self, ids: "ai") -> "aiai":
        return [[], []]

    @dbus_property(access=PropertyAccess.READ)
    def Version(self) -> "u":
        return 3

    @dbus_property(access=PropertyAccess.READ)
    def TextDirection(self) -> "s":
        return "ltr"

    @dbus_property(access=PropertyAccess.READ)
    def Status(self) -> "s":
        return "normal"

    @dbus_property(access=PropertyAccess.READ)
    def IconThemePath(self) -> "as":
        return []

    @signal()
    def LayoutUpdated(self) -> "ui":
        return [self.revision, 0]


class Tray:
    """Runs its own asyncio loop on a daemon thread. `on_command(str)` is called from it."""

    def __init__(self, on_command, state: str = "active"):
        self.on_command = on_command
        self.state = state
        self._loop = asyncio.new_event_loop()
        self._bus = None
        self._notif_id = 0
        threading.Thread(target=self._run, name="tray", daemon=True).start()

    def command(self, cmd: str) -> None:
        if cmd == "help":
            self.notify("Hand Control gestures", HELP, timeout=12000)
        else:
            self.on_command(cmd)

    # -- called from other threads ----------------------------------------------------
    def set_state(self, state: str) -> None:
        self._loop.call_soon_threadsafe(self._apply_state, state)

    def notify(self, title: str, body: str, timeout: int = 4000) -> None:
        asyncio.run_coroutine_threadsafe(self._notify(title, body, timeout), self._loop)

    # -- loop side --------------------------------------------------------------------
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._setup())
        except Exception as exc:  # no session bus / no tray host: run headless
            print(f"handctrl: tray unavailable ({exc})", flush=True)
            return
        self._loop.run_forever()

    async def _setup(self) -> None:
        self._bus = await MessageBus().connect()
        self._item, self._menu = _Item(self), _Menu(self)
        self._bus.export("/StatusNotifierItem", self._item)
        self._bus.export("/MenuBar", self._menu)
        name = f"org.kde.StatusNotifierItem-{os.getpid()}-1"
        await self._bus.request_name(name)
        intro = await self._bus.introspect("org.kde.StatusNotifierWatcher", "/StatusNotifierWatcher")
        watcher = self._bus.get_proxy_object(
            "org.kde.StatusNotifierWatcher", "/StatusNotifierWatcher", intro
        ).get_interface("org.kde.StatusNotifierWatcher")
        await watcher.call_register_status_notifier_item(name)

    def _apply_state(self, state: str) -> None:
        if state == self.state or self._bus is None:
            self.state = state
            return
        self.state = state
        self._item.NewIcon()
        self._item.NewToolTip()
        self._item.NewStatus("Active")
        self._menu.revision += 1
        self._menu.LayoutUpdated()

    async def _notify(self, title: str, body: str, timeout: int) -> None:
        if self._bus is None:
            print(f"handctrl: {title}: {body}", flush=True)
            return
        try:
            intro = await self._bus.introspect(
                "org.freedesktop.Notifications", "/org/freedesktop/Notifications"
            )
            n = self._bus.get_proxy_object(
                "org.freedesktop.Notifications", "/org/freedesktop/Notifications", intro
            ).get_interface("org.freedesktop.Notifications")
            self._notif_id = await n.call_notify(
                "Hand Control", self._notif_id, ICONS[self.state], title, body, [],
                {"transient": Variant("b", True)}, timeout,
            )
        except Exception:
            print(f"handctrl: {title}: {body}", flush=True)
