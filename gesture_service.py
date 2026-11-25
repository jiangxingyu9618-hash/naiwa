"""手势识别子进程：判定逻辑照抄 11-04 版 GestureWorker，只把执行位置搬出主进程"""
import argparse
import json
import math
import os
import shutil
import sys
import time


def _ascii_safe(path):
    """mediapipe 的 C++ 层打不开中文路径，含非 ASCII 时复制一份到公共目录再读"""
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    for base in (os.environ.get("PUBLIC"), os.path.join(os.environ.get("SystemDrive", "C:"), "ProgramData")):
        if not base:
            continue
        d = os.path.join(base, "YueXinMiao")
        try:
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, os.path.basename(path))
            if (not os.path.exists(dst)) or os.path.getsize(dst) != os.path.getsize(path):
                shutil.copyfile(path, dst)
            return dst
        except Exception:
            continue
    return path


def _d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _classify_hand(lm):
    """21个关键点 -> 规则手势: fist/open/pinch/none"""
    wrist = lm[0]
    if _d(lm[4], lm[8]) < 0.05:
        return "pinch"
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    extended = 0
    for t, p in zip(tips, pips):
        if _d(lm[t], wrist) > _d(lm[p], wrist) * 1.15:
            extended += 1
    if extended <= 1:
        return "fist"
    if extended >= 3:
        return "open"
    return "none"


def _is_v_sign(lm):
    """V字手（食指中指伸直，其他弯）= 切换模式信号"""
    wrist = lm[0]
    idx_ext = _d(lm[8], wrist) > _d(lm[6], wrist) * 1.15
    mid_ext = _d(lm[12], wrist) > _d(lm[10], wrist) * 1.15
    ring_bend = _d(lm[16], wrist) < _d(lm[14], wrist) * 1.15
    pinky_bend = _d(lm[20], wrist) < _d(lm[18], wrist) * 1.15
    return idx_ext and mid_ext and ring_bend and pinky_bend


def _build_vec(hands_list, np):
    """拼126维: 左手前63, 右手后63, 没手填0"""
    vec = np.zeros(126, dtype="float32")
    for hi, lm in enumerate(hands_list[:2]):
        if lm is None:
            continue
        off = hi * 63
        for i, p in enumerate(lm):
            vec[off + i * 3 + 0] = p.x
            vec[off + i * 3 + 1] = p.y
            vec[off + i * 3 + 2] = p.z
    return vec.reshape(1, 126)


_ONNX_MAP = {0: "heart1", 1: "heart2", 2: "ok", 3: "thumbsdown"}


def write_state(out, payload):
    tmp = out + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, out)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="hand_landmarker.task")
    ap.add_argument("--mlp", default="mlp_weights.npz")
    ap.add_argument("--mode", default="rule")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(base, args.model)
    model_path = _ascii_safe(model_path)
    mlp_path = args.mlp if os.path.isabs(args.mlp) else os.path.join(base, args.mlp)
    out = args.out if os.path.isabs(args.out) else os.path.join(base, args.out)
    mode_file = out + ".mode"
    stop_file = out + ".stop"
    dbg = out + ".debug.log"

    import cv2
    import numpy as np
    import mediapipe as mp

    mlp = None
    if os.path.exists(mlp_path):
        try:
            w = np.load(mlp_path)
            mlp = (w["network.0.weight"], w["network.0.bias"],
                   w["network.3.weight"], w["network.3.bias"],
                   w["network.6.weight"], w["network.6.bias"])
        except Exception as err:
            with open(dbg, "a", encoding="utf-8") as f:
                f.write("\n[mlp load fail] %s\n" % err)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    MImage = mp.Image
    MFormat = mp.ImageFormat

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=2,
        running_mode=RunningMode.VIDEO,
    )
    landmarker = HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    mode = args.mode
    if os.path.exists(mode_file):
        try:
            with open(mode_file, "r", encoding="utf-8") as f:
                mode = f.read().strip() or mode
        except OSError:
            pass

    last, streak, cooldown, ts = "none", 0, 0, 0
    seq = 0
    nframe = 0
    mean_val = 0.0
    hands_n = 0
    last_pinch_xy = None
    pinch_gap = 0
    t_start = time.time()

    try:
        while True:
            if os.path.exists(stop_file):
                break
            if nframe % 15 == 0 and os.path.exists(mode_file):
                try:
                    with open(mode_file, "r", encoding="utf-8") as f:
                        mode = f.read().strip() or mode
                except OSError:
                    pass
            nframe += 1
            if nframe == 40:
                t_start = time.time()  # 去掉模型加载时间，只算稳定后的帧率
            try:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                mean_val = float(frame.mean())
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = MImage(image_format=MFormat.SRGB, data=rgb)
                ts += 33
                res = landmarker.detect_for_video(mp_img, ts)
                hands = res.hand_landmarks if res.hand_landmarks else []
                hands_n = len(hands)
                g = "none"
                lm = None

                if mode == "rule":
                    if len(hands) == 1:
                        lm = hands[0]
                        if _is_v_sign(lm):
                            g = "none"
                        else:
                            g = _classify_hand(lm)
                else:
                    if len(hands) >= 2:
                        g = "heart2"
                        lm = hands[0]
                    elif len(hands) == 1:
                        lm = hands[0]
                        if _is_v_sign(lm):
                            g = "none"
                        elif mlp is not None:
                            x = _build_vec([lm, None], np)
                            W1, b1, W2, b2, W3, b3 = mlp
                            h = np.maximum(0, x @ W1.T + b1)
                            h = np.maximum(0, h @ W2.T + b2)
                            logits = h @ W3.T + b3
                            e = np.exp(logits - logits.max())
                            probs = e / e.sum()
                            cid = int(probs.argmax())
                            conf = float(probs.max())
                            if conf >= 0.8:
                                g = _ONNX_MAP.get(cid, "none")

                follow = False
                if g == "pinch" and lm is not None:
                    follow = True
                    last_pinch_xy = (float(lm[8].x), float(lm[8].y))
                    pinch_gap = 0
                elif last_pinch_xy is not None and pinch_gap < 5:
                    # 捏合偶尔抖掉几帧，仍沿用上一位置，避免拖动一顿一顿
                    follow = True
                    pinch_gap += 1

                if g == last:
                    streak += 1
                else:
                    last, streak = g, 0
                fire = bool(streak == 6 and g != "none" and g != "pinch" and cooldown <= 0 and not follow)
                if fire:
                    cooldown = 90
                if cooldown > 0:
                    cooldown -= 1

                seq += 1
                payload = {"seq": seq, "event": "pinch" if follow else g, "fire": fire, "mode": mode}
                if follow:
                    payload["x"] = last_pinch_xy[0]
                    payload["y"] = last_pinch_xy[1]
                write_state(out, payload)

                if nframe % 20 == 0:
                    fps = nframe / max(0.001, time.time() - t_start)
                    with open(dbg, "a", encoding="utf-8") as f:
                        f.write("f%d mean=%.1f hands=%d g=%s streak=%d cd=%d mode=%s fps=%.1f\n"
                                % (nframe, mean_val, hands_n, g, streak, cooldown, mode, fps))
            except Exception as frame_err:
                with open(dbg, "a", encoding="utf-8") as f:
                    f.write("[frame err] %s\n" % frame_err)
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            landmarker.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
