import cv2
import os
import numpy as np
from tqdm import tqdm
from utils import (
    overlay_original_preview,
    draw_right_bottom_text,
    four_in_one,
    masking,
    sixteen_in_one_pls_colormap,
    apply_edge_detection_by_region,
    difference_of_gaussian,
    morphological_operation_by_region,
    feature_detection_and_descriptor,
    retina_face,
    object_detection_yolo
)

INPUT_FILE = "video/sample.mp4"
base_dir = os.path.dirname(INPUT_FILE)
base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]

OUTPUT_FILE = os.path.join(base_dir, f"{base_name}_output.mp4")
os.makedirs(base_dir, exist_ok=True)


def main():
    cap = cv2.VideoCapture(INPUT_FILE)                      # Open the video file
    if not cap.isOpened():
        print("[ERROR] >> Cannot open the video file. Please check path, format, or codec support.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                # Define video codec
    # cap.set(cv2.CAP_PROP_POS_MSEC, 10000)                 # Start from x ms

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Constants for initial scaling animation (first N frames)
    FINAL_SCALE = 1.5
    SCALE_START_FRAME = 0
    SCALE_END_FRAME = 150

    position_order = list(range(16))  # for 16 in 1 colormap

    EFFECT_SCHEDULE = [
        (0.04, "[1] Keep the Size: 1.5x", lambda canvas, scaled, idx: canvas.copy()),
        (0.08, "[2] Rotation (3 turns)", lambda canvas, scaled, idx: cv2.warpAffine(
            canvas,
            cv2.getRotationMatrix2D(((canvas.shape[1]-1)/2.0, (canvas.shape[0]-1)/2.0),
                                    min(7.2 * (idx - SCALE_END_FRAME), 1080), 1),
            (canvas.shape[1], canvas.shape[0]))),
        (0.09, "[3] 4 in 1 + Masking", lambda canvas, scaled, idx: four_in_one(canvas, scaled, effects=[
            lambda x: masking(x, "circle"),
            lambda x: masking(x, "rectangle"),
            lambda x: masking(x, "ellipse"),
            lambda x: masking(x, "triangle")
        ])),
        (0.10, "[4] 4 in 1 + Flip", lambda canvas, scaled, idx: four_in_one(canvas, scaled, effects=[
            None,
            lambda x: cv2.flip(x, 1),
            lambda x: cv2.flip(x, 0),
            lambda x: cv2.flip(x, -1)
        ])),
        (0.07, "[5] 16 in 1 + ColorMap", lambda canvas, scaled, idx: sixteen_in_one_pls_colormap(
            canvas, scaled, idx, position_order)),
        (0.12, "[6] Edge Detection", lambda canvas, scaled, idx: apply_edge_detection_by_region(canvas)),
        (0.07, "[7] DoG", lambda canvas, scaled, idx: difference_of_gaussian(canvas)),
        (0.08, "[8] Morphological Operations", lambda canvas, scaled, idx: morphological_operation_by_region(canvas)),
        (0.09, "[9] Feature Detection & Descriptor", lambda canvas, scaled, idx: feature_detection_and_descriptor(canvas)),
        (0.09, "[10-1] RetinaFace + Mosaic", lambda canvas, scaled, idx: retina_face(canvas, mosaic=True)),
        (0.09, "[10-2] Face Detection (RetinaFace)", lambda canvas, scaled, idx: retina_face(canvas, mosaic=False)),
        (0.08, "[11] Object Detection (YOLOv11)", lambda canvas, scaled, idx: object_detection_yolo(canvas)),
    ]

    def build_effect_table(schedule, total_frames, start_offset):
        table = []
        start = start_offset
        for ratio, label, func in schedule:
            duration = int(total_frames * ratio)
            end = start + duration

            # Special handling for rotation to compute dynamic degrees
            if "Rotation" in label:
                degrees_per_frame = 1080 / duration
                wrapped_func = lambda canvas, scaled, idx, s=start, dpf=degrees_per_frame: cv2.warpAffine(
                    canvas,
                    cv2.getRotationMatrix2D(
                        ((canvas.shape[1] - 1) / 2.0, (canvas.shape[0] - 1) / 2.0),
                        (idx - s) * dpf, 1
                    ),
                    (canvas.shape[1], canvas.shape[0])
                )
                table.append((start, end, label, wrapped_func))
            else:
                table.append((start, end, label, func))
            start = end
        return table

    EFFECT_TABLE = build_effect_table(EFFECT_SCHEDULE, total_frames, SCALE_END_FRAME)

    out_w = int(w * FINAL_SCALE)
    out_h = int(h * FINAL_SCALE)
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (out_w, out_h))

    bar = tqdm(total=total_frames, desc="Processing frames")  # Initialize progress bar

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ## Stop when over x ms
        # if cap.get(cv2.CAP_PROP_POS_MSEC) > 75000:
        #     break
        
        original_frame = frame.copy()

        # --------------------------------------------
        # ------------ Modify Area -------------------
        # --------------------------------------------

        # === 1. Compute dynamic scaling ratio ===
        if frame_idx < SCALE_END_FRAME:
            alpha = frame_idx / SCALE_END_FRAME
            scale = 1.0 + alpha * (FINAL_SCALE - 1.0)   # 線性從 1.0 -> 1.5
        else:
            scale = FINAL_SCALE

        # === 2. Scale the original frame ===
        orig_h, orig_w = original_frame.shape[:2]
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        scaled_frame = cv2.resize(original_frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        offset_x = max((out_w - new_w) // 2, 0)
        offset_y = max((out_h - new_h) // 2, 0)
        crop_w = min(out_w, new_w)
        crop_h = min(out_h, new_h)
        canvas[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w] = scaled_frame[0:crop_h, 0:crop_w]

        processed_frame = canvas

        # === 3. Apply effect and overlay preview (after SCALE_END_FRAME) ===
        if frame_idx >= SCALE_END_FRAME:
            for start, end, label, func in EFFECT_TABLE:
                if start <= frame_idx < end:
                    effect_id = label
                    processed = func(processed_frame, scaled_frame, frame_idx)
                    if isinstance(processed, tuple):
                        processed_frame, position_order = processed
                    else:
                        processed_frame = processed
                    break
            else:
                effect_id = "[0] Original"
                processed_frame = canvas.copy()

            # Add original preview to top-left
            processed_frame = overlay_original_preview(
                base_frame=processed_frame,
                original_frame=scaled_frame,
                scale=0.25
            )
        else:
            effect_id = f"[0] Scaling to {scale:.2f}x"

        # Display caption on bottom-right
        status = f"{frame_idx}/{total_frames} frames, FPS: {fps:.1f}"
        processed_frame = draw_right_bottom_text(
            img=processed_frame,
            lines=[effect_id, status],
            margin=(25, 25)
        )

        out.write(processed_frame)
        bar.update(1)
        frame_idx += 1

    cap.release()
    out.release()
    bar.close()


if __name__ == "__main__":
    main()