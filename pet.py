import json
import os
import shutil
import subprocess
import sys

import PyQt5

# Qt 的插件路径从 Qt5Core.dll 位置推出来，路径含中文会被转成问号，
# 平台插件找不到、窗口起不来。非 ASCII 时复制插件到公共目录再指过去。
_QT_KEYS = ("platforms/qwindows.dll", "imageformats/qwebp.dll", "imageformats/qgif.dll")


def _qt_plugins():
    root = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    try:
        root.encode("ascii")
        return root
    except UnicodeEncodeError:
        pass
    base = os.environ.get("PUBLIC") or os.path.join(os.environ.get("SystemDrive", "C:"), "ProgramData")
    dst = os.path.join(base, "YueXinMiao", "qtplugins")
    if all(os.path.exists(os.path.join(dst, k)) for k in _QT_KEYS):
        return dst
    try:
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(root, dst)
        return dst
    except Exception:
        return root


qt_plugin_root = _qt_plugins()
os.environ["QT_PLUGIN_PATH"] = qt_plugin_root
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(qt_plugin_root, "platforms")

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QAction, QApplication, QLabel, QMenu, QWidget

BASE = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.path.join(BASE, "naiwa_frames")


def frame(name):
    return os.path.join(FRAME_DIR, name + ".webp")


IDLE_FRAME = frame("01_idle")
ACTION_FRAMES = {
    "laugh": frame("05_laugh"),
    "shy": frame("07_greet"),
    "cry": frame("06_cry"),
    "clap": frame("08_clap"),
    "cheer": frame("04_cheer"),
    "turn": frame("10_turn_out"),
}
ACTION_TITLES = [
    ("laugh", "笑"),
    ("shy", "害羞"),
    ("cry", "哭"),
    ("clap", "拍手"),
    ("cheer", "欢呼"),
    ("turn", "转身"),
]
GESTURE_ACTION = {
    "fist": "laugh",
    "open": "cry",
    "ok": "shy",
    "heart1": "clap",
    "heart2": "cheer",
    "thumbsdown": "turn",
}

CONFIG_PATH = os.path.join(BASE, "naiwa_config.json")
STATE_PATH = os.path.join(BASE, "gesture_state.json")
MODE_PATH = STATE_PATH + ".mode"
SERVICE_PATH = os.path.join(BASE, "gesture_service.py")
MODEL_PATH = os.path.join(BASE, "hand_landmarker.task")
MLP_PATH = os.path.join(BASE, "mlp_weights.npz")

WINDOW_W = 200
WINDOW_H = 220
FRAME_SIZE = 180
LOOPS_PER_ACTION = 3


class GestureProc(object):
    def __init__(self, script, state_path, model_path, mlp_path):
        self.script = script
        self.state_path = state_path
        self.model_path = model_path
        self.mlp_path = mlp_path
        self.stop_flag = state_path + ".stop"
        self.proc = None
        self.restarts = 0

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def raise_flag(self):
        try:
            with open(self.stop_flag, "w") as f:
                f.write("1")
        except Exception:
            pass

    def clear_flag(self):
        try:
            if os.path.isfile(self.stop_flag):
                os.remove(self.stop_flag)
        except Exception:
            pass

    def start(self, mode):
        try:
            if os.path.isfile(self.state_path):
                os.remove(self.state_path)
        except Exception:
            pass
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [sys.executable, self.script, "--out", self.state_path,
             "--model", self.model_path, "--mlp", self.mlp_path, "--mode", mode],
            cwd=BASE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )

    def kill_hard(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.proc = None

    def read(self):
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.cfg = self.load_config()
        screen = QApplication.primaryScreen().geometry()
        x = int(self.cfg.get("x", screen.width() - WINDOW_W - 40))
        y = int(self.cfg.get("y", screen.height() - WINDOW_H - 80))
        x = max(0, min(x, screen.width() - WINDOW_W))
        y = max(0, min(y, screen.height() - WINDOW_H))
        self.setGeometry(x, y, WINDOW_W, WINDOW_H)

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, WINDOW_W, WINDOW_H)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setAttribute(Qt.WA_TranslucentBackground)

        self.movies = []
        self.movie = None
        self.current = None
        self.loops = 0
        self.max_loops = None
        self.last_frame = -1

        self.drag_offset = None
        self.pinch_pos = None
        self.last_xy = None

        self.cv_mode = "rule"
        self.write_mode("rule")

        self.gesture = GestureProc(SERVICE_PATH, STATE_PATH, MODEL_PATH, MLP_PATH)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_gesture)
        self.last_seq = -1

        self.play_idle()
        QTimer.singleShot(0, self.keep_on_screen)
        if "--gesture" in sys.argv:
            QTimer.singleShot(1500, self.toggle_gesture)

    def load_config(self):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_config(self):
        data = self.load_config()
        data["x"] = self.x()
        data["y"] = self.y()
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def write_mode(self, mode):
        self.cv_mode = mode
        try:
            with open(MODE_PATH, "w", encoding="utf-8") as f:
                f.write(mode)
        except Exception:
            pass

    def set_movie(self, path, max_loops=None):
        if not os.path.isfile(path):
            return
        if self.movie is not None:
            self.movie.stop()
        movie = QMovie(path)
        movie.setCacheMode(QMovie.CacheAll)
        movie.setScaledSize(QSize(FRAME_SIZE, FRAME_SIZE))
        self.label.setMovie(movie)
        movie.frameChanged.connect(self.on_frame)
        movie.start()
        self.movie = movie
        self.movies.append(movie)
        if len(self.movies) > 12:
            self.movies = self.movies[-12:]
        self.current = path
        self.loops = 0
        self.max_loops = max_loops
        self.last_frame = -1

    def on_frame(self, n):
        prev = self.last_frame
        self.last_frame = n
        if self.max_loops is None:
            return
        if n == 0 and prev > 0:
            self.loops += 1
            if self.loops >= self.max_loops:
                self.max_loops = None
                current = self.movie
                QTimer.singleShot(0, lambda m=current: self.idle_if_current(m))

    def idle_if_current(self, movie):
        if self.movie is movie:
            self.play_idle()

    def play_idle(self):
        if self.current != IDLE_FRAME:
            self.set_movie(IDLE_FRAME, None)

    def show_action(self, key):
        path = ACTION_FRAMES.get(key)
        if path and os.path.isfile(path):
            self.set_movie(path, LOOPS_PER_ACTION)

    def keep_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = max(0, min(self.x(), screen.width() - WINDOW_W))
        y = max(0, min(self.y(), screen.height() - WINDOW_H))
        if x != self.x() or y != self.y():
            self.move(x, y)

    def follow_finger(self, nx, ny):
        screen = QApplication.primaryScreen().geometry()
        tx, ty = nx * screen.width(), ny * screen.height()
        if self.pinch_pos is None:
            self.pinch_pos = (tx, ty)
        else:
            sx, sy = self.pinch_pos
            self.pinch_pos = (sx * 0.15 + tx * 0.85, sy * 0.15 + ty * 0.85)
        px, py = self.pinch_pos
        nx_ = int(px - WINDOW_W / 2)
        ny_ = int(py - WINDOW_H / 2)
        if self.last_xy is not None and abs(nx_ - self.last_xy[0]) < 2 and abs(ny_ - self.last_xy[1]) < 2:
            return
        self.last_xy = (nx_, ny_)
        self.move(nx_, ny_)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self.drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_offset = None
            self.pinch_pos = None
            self.last_xy = None
            self.keep_on_screen()
            self.save_config()
            event.accept()

    def contextMenuEvent(self, event):
        self.make_menu().exec_(event.globalPos())

    def make_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: rgba(255, 255, 255, 245);
                border: 1px solid #B7E08A;
                border-radius: 8px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 26px;
                border-radius: 5px;
                color: #333;
                font-family: "Microsoft YaHei";
            }
            QMenu::item:selected {
                background-color: #B7E08A;
                color: white;
            }
        """)
        for key, title in ACTION_TITLES:
            action = QAction(title, self)
            action.triggered.connect(lambda _=False, k=key: self.show_action(k))
            menu.addAction(action)

        menu.addSeparator()

        cv_action = QAction("关闭手势控制" if self.gesture.alive() else "开启手势控制", self)
        cv_action.triggered.connect(self.toggle_gesture)
        menu.addAction(cv_action)

        mode_action = QAction("CV模式：规则（拳掌捏）" if self.cv_mode == "rule"
                              else "CV模式：MLP（比心/OK/倒赞）", self)
        mode_action.triggered.connect(self.toggle_cv_mode)
        menu.addAction(mode_action)

        menu.addSeparator()

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.quit_pet)
        menu.addAction(quit_action)
        return menu

    def toggle_gesture(self):
        if self.gesture.alive():
            self.stop_gesture()
        else:
            self.start_gesture()

    def start_gesture(self):
        self.gesture.kill_hard()
        self.gesture.clear_flag()
        self.gesture.restarts = 0
        self.write_mode(self.cv_mode)
        self.gesture.start(self.cv_mode)
        self.last_seq = -1
        self.pinch_pos = None
        self.last_xy = None
        self.timer.start(30)

    def stop_gesture(self):
        self.timer.stop()
        self.gesture.raise_flag()
        self.gesture.kill_hard()
        self.gesture.clear_flag()
        self.pinch_pos = None
        self.last_xy = None

    def toggle_cv_mode(self):
        self.write_mode("mlp" if self.cv_mode == "rule" else "rule")
        self.pinch_pos = None

    def poll_gesture(self):
        if not self.gesture.alive():
            if self.gesture.restarts < 3:
                self.gesture.restarts += 1
                self.gesture.clear_flag()
                self.write_mode(self.cv_mode)
                self.gesture.start(self.cv_mode)
                return
            self.timer.stop()
            return
        data = self.gesture.read()
        if not data:
            return
        seq = data.get("seq", 0)
        if seq == self.last_seq:
            return
        self.last_seq = seq
        event = data.get("event")
        if event in ("pinch", "follow"):
            self.follow_finger(float(data.get("x", 0.5)), float(data.get("y", 0.5)))
            return
        self.pinch_pos = None
        self.last_xy = None
        if data.get("fire") and event in GESTURE_ACTION:
            self.show_action(GESTURE_ACTION[event])

    def quit_pet(self):
        self.save_config()
        self.timer.stop()
        self.gesture.raise_flag()
        self.gesture.kill_hard()
        self.gesture.clear_flag()
        QApplication.quit()
        os._exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec_())
