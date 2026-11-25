# Naiwa

A yellow desktop frog. Six actions, driven by hand gestures from your webcam. Windows 10/11, x64.

## Run

Download `naiwa-portable-v1.0.0.zip` (208 MB) from [Releases](../../releases/tag/v1.0.0), unzip anywhere, double-click `启动.bat`. For gesture control, use `启动-手势.bat`.

No Python required. `runtime\` carries the interpreter and every dependency; deleting it breaks the app. Allow camera access when Windows asks, and close other apps that hold the webcam.

## Actions

Right-click menu: 笑 / 害羞 / 哭 / 拍手 / 欢呼 / 转身. Each action plays three loops and falls back to idle, and switching mid-action cuts straight to the new one.

Rule mode is the default.

| Gesture | Action |
| --- | --- |
| Fist | 笑 |
| Open palm | 哭 |
| Thumb-index pinch | Runs after your fingertip and turns to face it |

A one-hand heart does nothing on purpose.

MLP mode, switched from the right-click menu.

| Gesture | Action |
| --- | --- |
| Two-hand heart | 欢呼 |
| OK | 害羞 |
| Thumbs down | 转身 |

## From source

Python 3.11, 64-bit.

```
pip install -r requirements.txt
python pet.py
python pet.py --gesture
```

Recognition runs in a separate process (`gesture_service.py`); the main window polls its state file every 20 ms and restarts the service up to three times if it dies.

MIT
