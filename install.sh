#!/usr/bin/env bash
# Installs handctrl for the current user. Only the udev rule step needs sudo.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Installing Python environment (uv, Python 3.12)"
uv sync --no-dev
mkdir -p ~/.local/bin
ln -sf "$PWD/.venv/bin/handctrl" ~/.local/bin/handctrl

echo "==> Downloading hand model"
.venv/bin/python -c "from handctrl.tracker import ensure_model; print(ensure_model())"

if [ ! -w /dev/uinput ]; then
  echo "==> Allowing your user to create virtual input devices (sudo)"
  sudo install -m 644 packaging/60-handctrl-uinput.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules
  sudo udevadm trigger --name-match=uinput
fi

if ! /usr/bin/python3 -c "import PySide6" 2>/dev/null; then
  echo "   (optional) for the camera preview window: sudo dnf install python3-pyside6"
fi

echo "==> Enabling autostart (systemd user service)"
mkdir -p ~/.config/systemd/user
install -m 644 packaging/handctrl.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now handctrl.service

echo "Done. Tray icon: 'Hand Control'. Try: handctrl preview | handctrl calibrate | handctrl toggle"
