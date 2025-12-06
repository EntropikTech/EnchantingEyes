"""
视频处理器模块

处理视频文件，逐帧提取视线数据
"""

import cv2
import json
import mediapipe as mp
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from ..core.gaze_estimator import GazeEstimator
from ..visualization.drawer import GazeDrawer


class VideoProcessor:
    """视频处理器"""

    def __init__(
        self,
        gaze_estimator: Optional[GazeEstimator] = None,
        viz_level: int = 1
    ):
        """
        初始化视频处理器

        Args:
            gaze_estimator: 视线估算器实例，如果为None则创建默认实例
            viz_level: 可视化级别 (1=简单, 2=标准, 3=完整网格)
        """
        self.gaze_estimator = gaze_estimator or GazeEstimator()
        self.drawer = GazeDrawer(viz_level=viz_level)

        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # 视频模式，启用跟踪优化
            max_num_faces=1,
            refine_landmarks=True,    # 关键：启用虹膜检测
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_video(
        self,
        input_path: str,
        output_json_path: str,
        output_video_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理视频文件

        Args:
            input_path: 输入视频路径
            output_json_path: 输出JSON文件路径
            output_video_path: 输出可视化视频路径（可选）

        Returns:
            处理结果摘要
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {input_path}")

        # 打开视频
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化输出视频写入器
        video_writer = None
        if output_video_path:
            output_video_path = Path(output_video_path)
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )

        # 准备输出数据结构
        result = {
            "video_info": {
                "path": str(input_path),
                "fps": fps,
                "total_frames": total_frames,
                "resolution": [width, height]
            },
            "gaze_data": []
        }

        # 逐帧处理
        frame_idx = 0
        detected_count = 0

        with tqdm(total=total_frames, desc="处理视频") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 计算时间戳
                timestamp_ms = (frame_idx / fps) * 1000

                # 转换颜色空间 BGR -> RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                # 运行 Face Mesh 检测
                results = self.face_mesh.process(rgb_frame)

                # 处理检测结果
                frame_data = {
                    "frame": frame_idx,
                    "timestamp_ms": round(timestamp_ms, 2),
                    "face_detected": False,
                    "gaze": None,
                    "iris_3d": None,
                    "eyelid": None
                }

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # 估算视线
                    gaze_result = self.gaze_estimator.estimate_gaze(landmarks)

                    if gaze_result:
                        frame_data["face_detected"] = True
                        frame_data["gaze"] = gaze_result["averaged"]
                        frame_data["iris_3d"] = gaze_result["iris_3d"]
                        frame_data["eyelid"] = gaze_result.get("eyelid")
                        detected_count += 1

                        # 绘制可视化
                        if video_writer:
                            frame = self.drawer.draw_gaze(
                                frame, landmarks, gaze_result, (width, height)
                            )

                result["gaze_data"].append(frame_data)

                # 写入输出视频
                if video_writer:
                    video_writer.write(frame)

                frame_idx += 1
                pbar.update(1)

        # 释放资源
        cap.release()
        if video_writer:
            video_writer.release()

        # 保存JSON
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 返回处理摘要
        return {
            "total_frames": total_frames,
            "detected_frames": detected_count,
            "detection_rate": round(detected_count / total_frames * 100, 2),
            "output_json": str(output_json_path),
            "output_video": str(output_video_path) if output_video_path else None
        }

    def __del__(self):
        """释放资源"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
