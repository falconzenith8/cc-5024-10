"""生成演示数据: 已知真值变形的"参考图"与"扫描古地图"及地标文件。

输出到 demo/:
  reference.png            现代参考图(网格 + 道路 + 湖泊 + 地标十字)
  scanned.png              经已知变形场扭曲的"扫描古地图"(含局部纸张伸缩)
  landmarks.csv            正确地标 sx,sy,tx,ty (世界坐标=参考图像素坐标)
  landmarks_with_errors.csv  含 2 个错误地标(用于演示诊断)
用法: python3 scripts/make_demo_data.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georect.synth import forward_sample, smooth_field

W, H = 1100, 850          # 扫描图尺寸
RW, RH = 1000, 800        # 参考图尺寸
OUT = Path(__file__).resolve().parents[1] / "demo"


def make_reference():
    img = Image.new("RGB", (RW, RH), (245, 248, 250))
    dr = ImageDraw.Draw(img)
    for x in range(0, RW, 100):
        dr.line([(x, 0), (x, RH)], fill=(180, 190, 200), width=1)
    for y in range(0, RH, 100):
        dr.line([(0, y), (RW, y)], fill=(180, 190, 200), width=1)
    rng = np.random.default_rng(3)
    for _ in range(7):  # 道路
        pts = []
        x, y = rng.uniform(0, RW), rng.uniform(0, RH)
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(30):
            pts.append((x, y))
            ang += rng.uniform(-0.4, 0.4)
            x += 25 * np.cos(ang)
            y += 25 * np.sin(ang)
        dr.line(pts, fill=(120, 110, 100), width=3)
    for _ in range(5):  # 湖泊
        cx, cy = rng.uniform(100, RW - 100), rng.uniform(100, RH - 100)
        rx, ry = rng.uniform(25, 70), rng.uniform(20, 55)
        dr.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(150, 190, 220))
    return img


def true_field(pts):
    """真值变形: 全局仿射 + 两个平滑鼓包 + 一处局部纸张伸缩。"""
    base = (np.array([18.0, 12.0]), np.array([[1.02, 0.012], [-0.008, 0.985]]))
    bumps = [
        (300.0, 250.0, 230.0, 42.0, -25.0),   # 平滑鼓包
        (800.0, 620.0, 190.0, -35.0, 28.0),   # 平滑鼓包
        (640.0, 420.0, 60.0, 48.0, -38.0),    # 局部纸张伸缩(紧凑、强)
    ]
    return smooth_field((W, H), bumps, base=base)(pts)


def main():
    OUT.mkdir(exist_ok=True)
    ref = make_reference()
    ref.save(OUT / "reference.png")
    ref_arr = np.asarray(ref).astype(np.float64)

    scanned = forward_sample(true_field, ref_arr, W, H)
    scanned = np.clip(scanned, 0, 255)
    # 做旧: 纸张底色 + 噪声
    rng = np.random.default_rng(5)
    tint = np.array([232.0, 220, 195])
    scanned = 0.82 * scanned + 0.18 * tint
    scanned += rng.normal(0, 3.0, scanned.shape)
    scanned = np.clip(scanned, 0, 255).astype(np.uint8)
    Image.fromarray(scanned).save(OUT / "scanned.png")

    # 地标: 均匀网格 + 伸缩区加密
    xs = np.linspace(0.1 * W, 0.9 * W, 6)
    ys = np.linspace(0.1 * H, 0.9 * H, 5)
    gx, gy = np.meshgrid(xs, ys)
    src = np.column_stack([gx.ravel(), gy.ravel()])
    local = np.column_stack([
        rng.uniform(640 - 110, 640 + 110, 6),
        rng.uniform(420 - 110, 420 + 110, 6),
    ])
    src = np.vstack([src, local])
    dst = true_field(src)
    # 世界坐标 = 参考图像素坐标(y 向上): 参考图数组行向下, 故翻转 y
    dst_world = dst.copy()
    dst_world[:, 1] = RH - dst[:, 1]

    lines = ["sx,sy,tx,ty"]
    lines += [f"{s[0]:.1f},{s[1]:.1f},{t[0]:.2f},{t[1]:.2f}" for s, t in zip(src, dst_world)]
    (OUT / "landmarks.csv").write_text("\n".join(lines) + "\n")

    bad = dst_world.copy()
    bad[7] += np.array([90.0, -60.0])          # 错位地标
    bad[15], bad[16] = bad[16].copy(), bad[15].copy()  # 互换 -> 矛盾/折叠
    lines_bad = ["sx,sy,tx,ty"]
    lines_bad += [f"{s[0]:.1f},{s[1]:.1f},{t[0]:.2f},{t[1]:.2f}" for s, t in zip(src, bad)]
    (OUT / "landmarks_with_errors.csv").write_text("\n".join(lines_bad) + "\n")

    print(f"已生成 demo/ 数据: reference {RW}x{RH}, scanned {W}x{H}, 地标 {len(src)} 对")
    print("参考平面范围请设为: xmin=0 ymin=0 xmax=%d ymax=%d" % (RW, RH))


if __name__ == "__main__":
    main()
