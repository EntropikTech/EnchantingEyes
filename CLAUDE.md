# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

EnchantingEyes 是一个仿生眼实时视线模仿系统，通过 MediaPipe Face Mesh 捕捉人眼视线数据，并映射为机械眼球舵机控制指令。

## 常用命令

```powershell
# 环境安装
pip install -r requirements.txt
# 或使用 conda
conda env create -f environment.yml

# 视频处理（逐帧提取视线数据）
python src/main.py --input data/videos/test_0.mp4 --output_video --max_yaw 35 --max_pitch 25

# 图片处理（单张图片视线推理）
python src/infer.py --input face.jpg --output_image --show

# 批量测试
scripts/test_videos.sh
```

## 代码架构

三层设计：

```
应用层: main.py (视频) / infer.py (图片)
    ↓
处理层: processors/video_processor.py, processors/image_processor.py
    ↓
核心层: core/gaze_estimator.py (视线计算) + visualization/drawer.py (可视化)
```

**核心算法流程**：
1. MediaPipe Face Mesh 检测 478 个面部关键点（需 `refine_landmarks=True` 启用虹膜检测）
2. 提取虹膜中心点（左眼索引 468，右眼索引 473）
3. 计算虹膜相对眼眶边界的位置 → 映射为 Yaw/Pitch 角度

**关键索引定义**（`core/face_mesh_indices.py`）：
- 虹膜中心：LEFT_IRIS_CENTER=468, RIGHT_IRIS_CENTER=473
- 眼眶边界：用于计算虹膜相对位置

## 输出格式

JSON 结构包含：
- `video_info`: 视频元数据（fps, resolution, total_frames）
- `gaze_data`: 逐帧数据数组，每帧包含 `yaw`, `pitch`, `iris_3d` 坐标

## 目录结构

- `src/`: 核心源代码
- `data/videos/`: 测试视频（test_0/1/2.mp4）
- `results/`: 处理输出目录
- `experiments/`: 实验记录
