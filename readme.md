# VidPolyMorph

![Demo](./demo.gif)

## 簡介｜Introduction

VidPolyMorph 是一個使用 OpenCV、YOLO、RetinaFace 等技術實現的影片視覺特效展示工具。支援多種轉場效果與影像處理手法，結合 Docker 環境與 Jupyter Notebook 互動介面，可快速展示與輸出處理後影片。

VidPolyMorph is a visual effect showcase tool for video processing, based on OpenCV, YOLO, RetinaFace, and other techniques. It supports various morphing transitions and visual effects, with both Docker container and Jupyter Notebook support.

---

## 專案結構｜Project Structure

```
VidPolyMorph/
│
├── files/                          # 模型檔案 | Model weights
│   ├── yolo11m.pt
│   └── superpoint_v6_from_tf.pth
│
├── video/                          # 影片資料夾 | Input video
│   └── sample.mp4
│
├── utils/                          # 視覺處理模組 | Effect modules
│   └── *.py
│
├── .devcontainer/                 # Docker 設置 | Devcontainer settings
│   ├── Dockerfile
│   └── devcontainer.json
│
├── requirements.txt               # Python 相依套件 | Python dependencies
├── vid-poly-morph.ipynb           # Notebook 執行入口 | Jupyter interface
├── vid-poly-morph.py              # CLI 執行主程式 | Main CLI script
└── readme.md
```

---

## 安裝方式｜Installation

### 使用 Docker (Recommended)

1. 安裝 Docker Desktop 或 Docker Engine。
2. 使用 VSCode Remote Containers 開啟 `.devcontainer/`：
   * VSCode 自動會建構並啟動容器。
3. 或手動建構並執行容器：
```bash
docker build -t vidpolymorph .
docker run --gpus all -v ${PWD}:/workspace -w /workspace vidpolymorph python3 vid-poly-morph.py
```

---

## 執行方式｜How to Run

### 選擇一：使用 Notebook 互動操作
```bash
# 在容器內
jupyter notebook
```

開啟 `vid-poly-morph.ipynb`，修改參數或步驟後執行。

### 選擇二：直接執行 Python 腳本
```bash
python3 vid-poly-morph.py
```

執行後會讀取 `video/sample.mp4`，輸出特效合成影片到`video/sample_output.mp4`

---

## 支援特效｜Supported Effects

| Label | Description                  |
| :----------: | :---------------------------------: |
| `[0]`      | 放大原始影片至 1.5 倍 (Upscale to 1.5x) |
| `[1]`      | 固定放大倍率 (Keep Scaled View)         |
| `[2]`      | 旋轉三圈 (Rotate 3 Turns)             |
| `[3]`      | 4-in-1 with 遮罩 (Masking)          |
| `[4]`      | 4-in-1 with 翻轉 (Flip)             |
| `[5]`      | 16-in-1 with 彩色映射 (ColorMap)      |
| `[6]`      | 邊緣偵測 (Edge Detection)             |
| `[7]`      | DoG 高斯差分 (Difference of Gaussian) |
| `[8]`      | 形態學操作 (Morphological Operations)  |
| `[9]`      | 特徵點與描述子 (Feature + Descriptor)    |
| `[10-1]`   | RetinaFace 偵測 + 馬賽克 (RetinaFace Detection + Mosaic)   |
| `[10-2]`   | RetinaFace 偵測 (無馬賽克)   (RetinaFace Detection Only) |
| `[11]`     | YOLOv11 物件偵測（Object Detection with YOLOv11） |

---

## 套件需求｜Dependencies
>>（已於 Docker 中安裝）
+ 如需本地執行，安裝：
```bash
pip install -r requirements.txt
```

---

## 範例影片來源｜Sample Video Source
* **來源 / Source**: [Video by Huu Huynh from Pexels](https://www.pexels.com/video/17418615/)

---

## 附註說明｜Additional Notes

* 所有視覺效果模組均可於 `utils/` 中找到，結構模組化，方便擴充。
* 支援 GPU 加速（透過 `--gpus all`）。
* Notebook 與 Python 腳本功能完全等價，可自由選擇使用方式。

---

如需加入新特效，請新增模組於 `utils/` 並修改 `EFFECT_SCHEDULE` 清單。

---

## 外部資源｜External Dependencies
This project integrates the following third-party open-source tools:

* [RetinaFace](https://github.com/serengil/retinaface)
  → A high-accuracy face detector.

* [SuperPoint](https://github.com/rpautrat/SuperPoint)
  → A self-supervised interest point detector and descriptor.

---