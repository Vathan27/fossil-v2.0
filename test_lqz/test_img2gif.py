from pathlib import Path
from PIL import Image
import re

# ================= 配置区域 =================

# 图片所在文件夹
IMAGE_DIR = Path("/home/lqz27/research_ws/BarrGen/fossil/fossil-2.0/results/barr_dai_implements/plots_norm/")

# 图片匹配模式，例如 cegis_0000_lie.png
PATTERN = "cegis_0*_lie.png"

# 输出 GIF 文件名
OUTPUT_GIF = IMAGE_DIR / "cegis_lie.gif"

# 每帧持续时间，单位：毫秒
DURATION = 500

# 是否循环播放，0 表示无限循环
LOOP = 0

# ============================================


def iteration_number(path):
    """提取 CEGIS 迭代编号，用于排序。"""
    match = re.search(r"cegis_(\d+)_lie\.png$", path.name)

    if match:
        return int(match.group(1))

    return float("inf")


def make_gif():
    # 搜索图片并按迭代编号排序
    image_files = sorted(
        IMAGE_DIR.glob(PATTERN),
        key=iteration_number
    )

    if not image_files:
        print(f"未找到匹配的图片：{IMAGE_DIR / PATTERN}")
        return

    print(f"找到 {len(image_files)} 张图片：")

    for path in image_files:
        print(f"  {path.name}")

    # 读取图片
    frames = []

    for path in image_files:
        with Image.open(path) as img:
            frames.append(img.convert("RGB").copy())

    # 检查图片尺寸是否一致
    sizes = {frame.size for frame in frames}

    # if len(sizes) != 1:
    #     print("错误：图片尺寸不一致，请先统一图片尺寸。")

    #     for frame in frames:
    #         frame.close()

    #     return

    # 保存 GIF
    frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=LOOP,
        optimize=True,
    )

    for frame in frames:
        frame.close()

    print(f"\nGIF 已生成：{OUTPUT_GIF.resolve()}")
    print(f"总帧数：{len(image_files)}")
    print(f"每帧时长：{DURATION} ms")


if __name__ == "__main__":
    make_gif()
