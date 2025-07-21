import cv2

def draw_right_bottom_text(img, lines, fontface=cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale=1.55, line_type=cv2.LINE_AA,
                           margin=(15,10), line_gap=20):
    """
    在影像右下角畫出靠右對齊的多行文字。
    lines: List of string - 每一行的內容
    margin: (x_margin, y_margin) - 右下角的邊界
    """
    FONT_COLOR = (255,206,85)
    OUTLINE_COLOR = (30,30,30)

    h_img, w_img = img.shape[:2]
    x_margin, y_margin = margin

    # 先預估整體高度
    line_sizes = [cv2.getTextSize(line, fontface, font_scale, thickness=4)[0] for line in lines]
    total_height = sum(size[1] for size in line_sizes) + line_gap * (len(lines) - 1)

    # 起始的 baseline Y 座標（從下往上疊）
    y_start = h_img - y_margin - total_height + line_sizes[0][1]


    for i, (text, (text_w, text_h)) in enumerate(zip(lines, line_sizes)):
        x = w_img - x_margin - text_w   # 靠右對齊
        y = y_start + i * (text_h + line_gap)
        cv2.putText(img, text, (x, y), fontface, font_scale, OUTLINE_COLOR, thickness=7, lineType=line_type)

    for i, (text, (text_w, text_h)) in enumerate(zip(lines, line_sizes)):
        x = w_img - x_margin - text_w
        y = y_start + i * (text_h + line_gap)
        cv2.putText(img, text, (x, y), fontface, font_scale, FONT_COLOR, thickness=4, lineType=line_type)

    return img
