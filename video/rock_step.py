"""Генерирует обучающее motion-видео «Бачата: рок-степ назад».

Слева — фигура танцора (вид сбоку), справа — следы стоп (вид сверху),
внизу — счёт 1–8 и подсказка. Звук — синтезированный ритм (гуира, бонго, бас).

Запуск:  python3 video/rock_step.py  [путь_к_mp4]
Нужны: Pillow, numpy, ffmpeg.
"""
import math
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, SS, FPS = 1280, 720, 2, 30
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BG = (17, 15, 27)
PANEL = (27, 24, 41)
LINE = (56, 51, 80)
TEXT = (238, 234, 248)
MUTED = (150, 143, 175)
BODY = (205, 199, 230)
LC = (79, 195, 247)   # левая нога
RC = (255, 138, 101)  # правая нога
ACC = (255, 213, 79)

_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_B if bold else FONT_R, int(size * SS))
    return _fonts[key]


def P(x, y):
    return (x * SS, y * SS)


def clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))


def ease(s):
    s = clamp(s)
    return s * s * (3 - 2 * s)


def lerp(a, b, s):
    return a + (b - a) * s


def shade(c, k):
    return tuple(int(v * k) for v in c[:3])


def rgba(c, a):
    return (c[0], c[1], c[2], int(clamp(a, 0, 255)))


# ---------------------------------------------------------------- хореография
STEP = 0.35  # длина шага назад, м
LAT = 0.11   # половина ширины постановки стоп, м

# счёт -> (x левой, x правой, доля веса на левой); x — вперёд/назад, м
POSES = {
    1: (-STEP, 0.0, 1.0),
    2: (-STEP, 0.0, 0.0),
    3: (0.0, 0.0, 1.0),
    4: (0.0, 0.0, 1.0),
    5: (0.0, -STEP, 0.0),
    6: (0.0, -STEP, 1.0),
    7: (0.0, 0.0, 0.0),
    8: (0.0, 0.0, 0.0),
}
TAP = {4: "R", 8: "L"}
ACTOR = {1: "L", 2: "R", 3: "L", 4: "R", 5: "R", 6: "L", 7: "R", 8: "L"}
DESC = {
    1: "Левая — шаг назад на носок",
    2: "Перенос веса вперёд на правую (рок)",
    3: "Левая приставляется к правой",
    4: "Тап правой, акцент бедром влево",
    5: "Правая — шаг назад на носок",
    6: "Перенос веса вперёд на левую (рок)",
    7: "Правая приставляется к левой",
    8: "Тап левой, акцент бедром вправо",
}

TITLE_DUR = 3.0
OUTRO_DUR = 5.0


def build_events():
    events = []
    t = TITLE_DUR + 0.4
    phases = [("Медленно", 80, 2), ("В темпе музыки", 128, 3)]
    for pi, (name, bpm, loops) in enumerate(phases):
        T = 60.0 / bpm
        for c in (5, 6, 7, 8):
            w = 0.0 if (c == 8 or pi > 0) else 0.5
            events.append(dict(t=t, T=T, count=c, intro=True, pose=(0.0, 0.0, w),
                               tap=None, phase=name, bpm=bpm))
            t += T
        for _ in range(loops):
            for c in range(1, 9):
                events.append(dict(t=t, T=T, count=c, intro=False, pose=POSES[c],
                                   tap=TAP.get(c), phase=name, bpm=bpm))
                t += T
    return events


EVENTS = build_events()
DANCE_END = EVENTS[-1]["t"] + 1.2
TOTAL = DANCE_END + OUTRO_DUR
START_POSE = (0.0, 0.0, 0.5)


def window(e):
    return 0.55 * e["T"]


def state(t):
    """Поза в момент t: позиции стоп, подъём, вес, акцент бедра, активный счёт."""
    prev = START_POSE
    active = None
    for i, e in enumerate(EVENTS):
        if t >= e["t"] - window(e):
            active = i
    st = dict(active=active)
    for e in EVENTS:
        start = e["t"] - window(e)
        if t < start:
            break
        if t < e["t"]:
            s = (t - start) / window(e)
            p1 = e["pose"]
            k = ease(s)
            st["Lx"], st["Rx"] = lerp(prev[0], p1[0], k), lerp(prev[1], p1[1], k)
            st["wL"] = lerp(prev[2], p1[2], ease((s - 0.25) / 0.75))
            arc = math.sin(math.pi * clamp(s))
            st["liftL"] = 0.07 * arc if prev[0] != p1[0] else 0.0
            st["liftR"] = 0.07 * arc if prev[1] != p1[1] else 0.0
            if e["tap"] == "L":
                st["liftL"] = 0.05 * arc
            if e["tap"] == "R":
                st["liftR"] = 0.05 * arc
            st["moving"] = (prev, p1, s)
            break
        prev = e["pose"]
    else:
        pass
    if "Lx" not in st:
        st.update(Lx=prev[0], Rx=prev[1], wL=prev[2], liftL=0.0, liftR=0.0, moving=None)
    # акцент бедра на тапе — в сторону опорной ноги
    pop = 0.0
    for e in EVENTS:
        if e["tap"]:
            sign = -1.0 if e["tap"] == "R" else 1.0
            pop += sign * 0.05 * math.exp(-((t - e["t"]) / (0.22 * e["T"])) ** 2)
    st["pop"] = pop
    return st


# ---------------------------------------------------------------- вид сбоку
SIDE_X0, FLOOR_Y, SCALE = 380, 545, 250
ANK = 0.08
THIGH, SHIN = 0.46, 0.45


def sp(x, y):
    return P(SIDE_X0 + x * SCALE, FLOOR_Y - y * SCALE)


def rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


FOOT_SIDE = [(-0.18, 0.0), (-0.195, 0.045), (-0.14, 0.095), (-0.04, 0.065),
             (0.05, 0.03), (0.065, 0.0)]  # относительно подушечки стопы


def leg_geometry(hip, ax, lift):
    ball = (ax + 0.13, lift)
    v = (-0.13, ANK)
    reach = THIGH + SHIN - 0.006
    theta = 0.0
    def ankle_at(th):
        r = rot(v, -th)
        return (ball[0] + r[0], ball[1] + r[1])
    if math.dist(hip, ankle_at(0.0)) > reach:
        lo, hi = 0.0, 1.1
        for _ in range(30):
            mid = (lo + hi) / 2
            if math.dist(hip, ankle_at(mid)) > reach:
                lo = mid
            else:
                hi = mid
        theta = hi
    ankle = ankle_at(theta)
    d = min(math.dist(hip, ankle), THIGH + SHIN - 1e-4)
    phi = math.atan2(ankle[1] - hip[1], ankle[0] - hip[0])
    a = math.acos(clamp((THIGH ** 2 + d ** 2 - SHIN ** 2) / (2 * THIGH * d), -1, 1))
    knee = (hip[0] + THIGH * math.cos(phi + a), hip[1] + THIGH * math.sin(phi + a))
    foot = [(ball[0] + r[0], ball[1] + r[1]) for r in (rot(p, -theta) for p in FOOT_SIDE)]
    return knee, ankle, foot


def seg(d, a, b, w, col):
    d.line([sp(*a), sp(*b)], fill=col, width=int(w * SCALE * SS))
    r = w * SCALE * SS / 2
    for p in (a, b):
        x, y = sp(*p)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)


def draw_side(d, st):
    Lx, Rx, wL = st["Lx"], st["Rx"], st["wL"]
    hx = wL * Lx + (1 - wL) * Rx + 0.02
    hy = 0.93 - 0.025 * abs(Lx - Rx) / STEP
    hip = (hx, hy)
    # тень
    cx, cy = sp(hx, 0)
    d.ellipse([cx - 95 * SS, cy - 9 * SS, cx + 95 * SS, cy + 9 * SS], fill=(0, 0, 0, 90))

    shoulder = (hx + 0.03, hy + 0.52)
    elbow = (shoulder[0] + 0.05, shoulder[1] - 0.26)
    hand = (elbow[0] + 0.25, elbow[1] + 0.06)

    # дальняя сторона: правая нога и правая рука
    far_body = shade(BODY, 0.62)
    seg(d, shoulder, (elbow[0] - 0.02, elbow[1]), 0.07, far_body)
    seg(d, (elbow[0] - 0.02, elbow[1]), (hand[0] - 0.03, hand[1] + 0.02), 0.06, far_body)
    FAR = 0.045  # параллакс: дальняя нога чуть смещена, чтобы её было видно
    far_hip = (hip[0] - FAR, hip[1] + 0.01)
    rk, ra, rf = leg_geometry(far_hip, Rx - FAR, st["liftR"])
    rcol = shade(RC, 0.85)
    seg(d, far_hip, rk, 0.085, rcol)
    seg(d, rk, ra, 0.075, rcol)
    d.polygon([sp(*p) for p in rf], fill=shade(RC, 0.6))

    # корпус
    seg(d, hip, shoulder, 0.15, BODY)
    neck = (shoulder[0] + 0.01, shoulder[1] + 0.07)
    seg(d, shoulder, neck, 0.06, BODY)
    hc = (shoulder[0] + 0.025, shoulder[1] + 0.18)
    x, y = sp(*hc)
    r = 0.105 * SCALE * SS
    d.ellipse([x - r, y - r, x + r, y + r], fill=BODY)
    # нос/взгляд вперёд
    nx, ny = sp(hc[0] + 0.1, hc[1] - 0.005)
    d.ellipse([nx - 6 * SS, ny - 6 * SS, nx + 6 * SS, ny + 6 * SS], fill=BODY)

    # ближняя сторона: левая нога и левая рука
    lk, la, lf = leg_geometry(hip, Lx, st["liftL"])
    seg(d, hip, lk, 0.09, LC)
    seg(d, lk, la, 0.08, LC)
    d.polygon([sp(*p) for p in lf], fill=shade(LC, 0.7))
    seg(d, shoulder, elbow, 0.075, BODY)
    seg(d, elbow, hand, 0.065, BODY)



# ---------------------------------------------------------------- вид сверху
TOP_CX, TOP_CY, TSCALE = 960, 300, 380

FOOT_TOP_R = [(0.000, -0.065), (0.025, -0.058), (0.035, -0.035), (0.036, 0.0),
              (0.040, 0.05), (0.046, 0.10), (0.046, 0.14), (0.035, 0.170),
              (0.015, 0.19), (-0.010, 0.205), (-0.035, 0.195), (-0.045, 0.16),
              (-0.043, 0.12), (-0.030, 0.07), (-0.022, 0.03), (-0.030, -0.01),
              (-0.030, -0.04), (-0.020, -0.06)]
TOES_R = [(-0.028, 0.226, 0.014), (-0.004, 0.224, 0.0095), (0.014, 0.217, 0.0085),
          (0.029, 0.206, 0.0075), (0.041, 0.191, 0.0065)]


def tp(lat, fwd):
    return P(TOP_CX + lat * TSCALE, TOP_CY - fwd * TSCALE)


def draw_footprint(d, side, fx, fill_a, col, lift=0.0, outline_a=255, dashed=False):
    lat0 = -LAT if side == "L" else LAT
    m = -1 if side == "L" else 1
    k = 1.0 + 2.2 * lift
    def pt(u, v):
        return tp(lat0 + m * u * k, fx + v * k)
    if lift > 0.005:
        sh = 4 + 120 * lift
        pts = [pt(u, v) for u, v in FOOT_TOP_R]
        d.polygon([(x + sh * SS, y + sh * SS) for x, y in pts], fill=(0, 0, 0, 70))
    poly = [pt(u, v) for u, v in FOOT_TOP_R]
    d.polygon(poly, fill=rgba(col, fill_a))
    if dashed:
        for i in range(0, len(poly), 2):
            d.line([poly[i], poly[(i + 1) % len(poly)]], fill=rgba(col, outline_a), width=2 * SS)
    else:
        d.line(poly + [poly[0]], fill=rgba(col, outline_a), width=int(2.5 * SS), joint="curve")
    for u, v, r in TOES_R:
        x, y = pt(u, v)
        rr = r * k * TSCALE * SS
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=rgba(col, max(fill_a, outline_a * 0.5)))
    return pt(0.0, 0.07)


def arrow(d, a, b, col, alpha, width=4):
    (x1, y1), (x2, y2) = a, b
    ang = math.atan2(y2 - y1, x2 - x1)
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 1:
        return
    hl = 16 * SS
    xe, ye = x2 - math.cos(ang) * hl * 0.8, y2 - math.sin(ang) * hl * 0.8
    d.line([a, (xe, ye)], fill=rgba(col, alpha), width=width * SS)
    left = (x2 - hl * math.cos(ang - 0.45), y2 - hl * math.sin(ang - 0.45))
    right = (x2 - hl * math.cos(ang + 0.45), y2 - hl * math.sin(ang + 0.45))
    d.polygon([(x2, y2), left, right], fill=rgba(col, alpha))


def draw_top(d, t, st):
    # возможные позиции — бледные контуры
    for side, col in (("L", LC), ("R", RC)):
        for fx in (0.0, -STEP):
            draw_footprint(d, side, fx, 0, col, outline_a=45, dashed=True)

    ev = EVENTS[st["active"]] if st["active"] is not None else None
    # стрелка движения активного счёта
    if ev is not None and not ev["intro"]:
        i = st["active"]
        prev = EVENTS[i - 1]["pose"] if i > 0 else START_POSE
        cur = ev["pose"]
        fade = clamp(1.0 - (t - ev["t"]) / (0.5 * ev["T"])) if t > ev["t"] else 1.0
        for idx, side, col in ((0, "L", LC), (1, "R", RC)):
            if prev[idx] != cur[idx]:
                lat = -LAT if side == "L" else LAT
                lat += -0.085 if side == "L" else 0.085
                a = tp(lat, prev[idx] + 0.07)
                b = tp(lat, cur[idx] + 0.07)
                arrow(d, a, b, col, 230 * fade)

    wL = st["wL"]
    order = [("L", LC, st["Lx"], st["liftL"], wL), ("R", RC, st["Rx"], st["liftR"], 1 - wL)]
    order.sort(key=lambda o: o[3])  # поднятая стопа рисуется поверх
    centers = {}
    for side, col, fx, lift, w in order:
        c = draw_footprint(d, side, fx, 55 + 190 * w, col, lift=lift)
        centers[side] = c
        x, y = c
        d.text((x, y), "Л" if side == "L" else "П", font=font(17, True),
               fill=(20, 18, 30) if w > 0.5 else rgba(col, 255), anchor="mm")

    # центр таза над стопами
    hip_lat = wL * (-LAT) + (1 - wL) * LAT + st["pop"]
    hip_fwd = wL * st["Lx"] + (1 - wL) * st["Rx"] + 0.07
    x, y = tp(hip_lat, hip_fwd)
    rx, ry = 0.19 * TSCALE * SS, 0.075 * TSCALE * SS
    d.ellipse([x - rx, y - ry, x + rx, y + ry], outline=(255, 255, 255, 150), width=3 * SS)
    d.ellipse([x - 7 * SS, y - 7 * SS, x + 7 * SS, y + 7 * SS], fill=(255, 255, 255, 220))

    # кольцо тапа
    for e in EVENTS:
        if e["tap"] and 0 <= t - e["t"] < 0.45:
            k = (t - e["t"]) / 0.45
            col = LC if e["tap"] == "L" else RC
            fx = e["pose"][0] if e["tap"] == "L" else e["pose"][1]
            lat = -LAT if e["tap"] == "L" else LAT
            cx, cy = tp(lat, fx + 0.17)
            r = (14 + 46 * k) * SS
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=rgba(col, 255 * (1 - k)), width=4 * SS)
            d.text((cx, cy - r - 12 * SS), "тап!", font=font(18, True),
                   fill=rgba(col, 255 * (1 - k)), anchor="mm")

    # шкала веса
    bx0, bx1, by = 830, 1090, 520
    d.text(P((bx0 + bx1) / 2, by - 20), "вес тела", font=font(14), fill=MUTED, anchor="mm")
    split = bx0 + (bx1 - bx0) * wL
    d.rounded_rectangle([*P(bx0, by - 7), *P(bx1, by + 7)], radius=7 * SS, fill=rgba(RC, 255))
    if split > bx0 + 1:
        d.rounded_rectangle([*P(bx0, by - 7), *P(max(split, bx0 + 14), by + 7)],
                            radius=7 * SS, fill=rgba(LC, 255))
    d.text(P(bx0 - 16, by), "Л", font=font(16, True), fill=LC, anchor="mm")
    d.text(P(bx1 + 16, by), "П", font=font(16, True), fill=RC, anchor="mm")


# ---------------------------------------------------------------- статичный фон
def make_background():
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([*P(30, 95), *P(640, 560)], radius=18 * SS, fill=PANEL)
    d.rounded_rectangle([*P(660, 95), *P(1250, 560)], radius=18 * SS, fill=PANEL)
    d.text(P(50, 115), "ВИД СБОКУ", font=font(14, True), fill=MUTED, anchor="lm")
    d.text(P(680, 115), "ВИД СВЕРХУ", font=font(14, True), fill=MUTED, anchor="lm")
    d.ellipse([*P(1092, 109), *P(1118, 121)], outline=(255, 255, 255, 150), width=2 * SS)
    d.text(P(1126, 115), "центр таза", font=font(13), fill=MUTED, anchor="lm")
    for i, (col, lbl) in enumerate(((LC, "левая нога"), (RC, "правая нога"))):
        y = 142 + i * 22
        d.ellipse([*P(50, y - 6), *P(62, y + 6)], fill=col)
        d.text(P(70, y), lbl, font=font(13), fill=MUTED, anchor="lm")
    # пол
    d.line([sp(-1.25, 0), sp(1.0, 0)], fill=LINE, width=3 * SS)
    for x, lbl in ((0.0, "исходная"), (-STEP, "шаг назад")):
        px, py = sp(x + 0.13, 0)
        d.line([(px, py), (px, py + 10 * SS)], fill=MUTED, width=2 * SS)
        d.text((px, py + 22 * SS), lbl, font=font(13), fill=MUTED, anchor="mm")
    x0, y0 = sp(-1.1, -0.05)
    d.text((x0, y0), "← назад", font=font(13), fill=MUTED, anchor="lm")
    x1, y1 = sp(0.95, -0.05)
    d.text((x1, y1), "вперёд →", font=font(13), fill=MUTED, anchor="rm")
    # направление взгляда в виде сверху
    ax, ay = tp(0.0, 0.42)
    d.text((ax, ay - 2 * SS), "↑ лицом сюда", font=font(13), fill=MUTED, anchor="mm")
    # заголовок
    d.text(P(32, 40), "Бачата · Рок-степ", font=font(32, True), fill=TEXT, anchor="lm")
    d.text(P(34, 74), "базовый шаг с рок-степом назад (back rock step)",
           font=font(16), fill=MUTED, anchor="lm")
    return img


def draw_hud(d, t, st):
    ev = EVENTS[st["active"]] if st["active"] is not None else EVENTS[0]
    # фаза и темп
    label = f"{ev['phase']} · {ev['bpm']} BPM"
    f = font(17, True)
    tw = d.textlength(label, font=f) / SS
    d.rounded_rectangle([*P(1250 - tw - 36, 24), *P(1250, 60)], radius=18 * SS,
                        fill=(255, 213, 79, 40), outline=rgba(ACC, 200), width=2 * SS)
    d.text(P(1250 - tw / 2 - 18, 42), label, font=f, fill=ACC, anchor="mm")

    beat_t = ev["t"]
    since = t - beat_t
    pulse = math.exp(-max(since, 0) * 7) if since >= 0 else 0.0

    # описание
    if st["active"] is None or ev["intro"]:
        desc, col = "Приготовились…  вес на правой, старт с левой", MUTED
        if st["active"] is not None:
            desc = f"Вступление: {ev['count']}…  старт с левой ноги"
    else:
        desc = DESC[ev["count"]]
        col = LC if ACTOR[ev["count"]] == "L" else RC
    d.text(P(640, 596), desc, font=font(26, True), fill=col if not ev["intro"] else TEXT,
           anchor="mm")

    # счёт 1..8
    for i in range(1, 9):
        cx, cy = 640 + (i - 4.5) * 72, 662
        acol = LC if ACTOR[i] == "L" else RC
        is_active = st["active"] is not None and not ev["intro"] and ev["count"] == i
        r = 24
        if is_active:
            r = 24 * (1 + 0.28 * pulse)
            d.ellipse([*P(cx - r, cy - r), *P(cx + r, cy + r)], fill=rgba(acol, 255))
            d.text(P(cx, cy), str(i), font=font(22, True), fill=(20, 18, 30), anchor="mm")
        else:
            d.ellipse([*P(cx - r, cy - r), *P(cx + r, cy + r)], outline=rgba(acol, 110),
                      width=2 * SS, fill=(255, 255, 255, 8))
            d.text(P(cx, cy), str(i), font=font(20, True), fill=rgba(acol, 170), anchor="mm")
        if i in (4, 8):
            d.text(P(cx, cy + 34), "тап", font=font(12), fill=MUTED, anchor="mm")

    # большой номер вступления
    if st["active"] is not None and ev["intro"] and since > -0.05:
        a = 255 * clamp(1 - since / ev["T"])
        sz = int(64 + 30 * pulse)
        d.text(P(560, 170), str(ev["count"]), font=font(sz, True), fill=rgba(ACC, a), anchor="mm")


def overlay_card(img, alpha, lines):
    if alpha <= 0:
        return
    ov = Image.new("RGBA", img.size, (*BG, int(248 * alpha)))
    d = ImageDraw.Draw(ov)
    for y, text, size, bold, col in lines:
        d.text(P(640, y), text, font=font(size, bold), fill=rgba(col, 255 * alpha), anchor="mm")
    img.paste(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))


TITLE_LINES = [
    (250, "Бачата", 30, False, MUTED),
    (315, "Рок-степ назад", 58, True, TEXT),
    (385, "8 счётов · сначала медленно, потом в темпе музыки", 20, False, MUTED),
    (450, "Л — левая нога      П — правая нога", 20, True, ACC),
]
OUTRO_LINES = [
    (170, "Схема рок-степа", 40, True, TEXT),
    (250, "1 — Л назад   ·   2 — вес на П   ·   3 — Л к П   ·   4 — тап П", 22, True, LC),
    (300, "5 — П назад   ·   6 — вес на Л   ·   7 — П к Л   ·   8 — тап Л", 22, True, RC),
    (385, "Шаг назад — на подушечку стопы, колени мягкие, корпус ровно", 19, False, TEXT),
    (425, "Вес переносится полностью: на тапе опорная нога держит весь вес", 19, False, TEXT),
    (465, "Партнёрша выполняет зеркально — начинает с правой ноги", 19, False, TEXT),
    (540, "Повторяйте медленно, пока шаги не станут автоматическими", 17, False, MUTED),
]


def render_frame(bg, t):
    img = bg.copy()
    d = ImageDraw.Draw(img, "RGBA")
    st = state(t)
    draw_side(d, st)
    draw_top(d, t, st)
    draw_hud(d, t, st)
    if t < TITLE_DUR:
        overlay_card(img, 1.0 - ease((t - (TITLE_DUR - 0.6)) / 0.6), TITLE_LINES)
    if t > DANCE_END:
        overlay_card(img, ease((t - DANCE_END) / 0.7), OUTRO_LINES)
    return img.reduce(SS)


# ---------------------------------------------------------------- звук
SR = 44100


def synth_audio(path):
    n = int(TOTAL * SR) + SR
    buf = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)

    def add(t0, sig):
        i = int(t0 * SR)
        j = min(n, i + len(sig))
        buf[i:j] += sig[: j - i]

    def tone(f0, f1, dur, decay, amp):
        tt = np.arange(int(dur * SR)) / SR
        f = f1 + (f0 - f1) * np.exp(-tt * 30)
        ph = 2 * np.pi * np.cumsum(f) / SR
        return (amp * np.sin(ph) * np.exp(-tt / decay)).astype(np.float32)

    def guira(dur, amp):
        m = int(dur * SR)
        x = rng.standard_normal(m).astype(np.float32)
        x = np.diff(np.concatenate([[0], x]))  # высокие частоты
        tt = np.arange(m) / SR
        env = np.minimum(tt / 0.008, 1) * np.exp(-tt / (dur * 0.35))
        return (amp * 0.25 * x * env).astype(np.float32)

    for e in EVENTS:
        t0, T, c = e["t"], e["T"], e["count"]
        if e["intro"]:
            add(t0, tone(1500, 1400, 0.06, 0.015, 0.5))
            continue
        add(t0, guira(min(0.18, T * 0.45), 0.9))
        add(t0 + T / 2, guira(min(0.12, T * 0.3), 0.5))
        if c in (4, 8):
            add(t0, tone(180, 70, 0.4, 0.12, 0.9))
            add(t0, tone(330, 300, 0.2, 0.06, 0.45))
        else:
            add(t0, tone(520, 440, 0.15, 0.045, 0.45 if c not in (1, 5) else 0.6))
            add(t0, tone(110, 90, 0.25, 0.08, 0.35))
    buf /= max(1e-6, np.abs(buf).max()) / 0.85
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "rock_step.mp4")
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "beat.wav")
        synth_audio(wav)
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-preset", "medium", "-c:a", "aac", "-b:a", "160k", "-shortest",
               "-movflags", "+faststart", out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        bg = make_background()
        frames = int(TOTAL * FPS)
        for f in range(frames):
            proc.stdin.write(render_frame(bg, f / FPS).tobytes())
            if f % 150 == 0:
                print(f"кадр {f}/{frames}", flush=True)
        proc.stdin.close()
        proc.wait()
    print("готово:", out, f"({TOTAL:.1f} с)")


if __name__ == "__main__":
    main()
