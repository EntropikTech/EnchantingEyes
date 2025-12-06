"""
视线估算器模块

基于虹膜位置相对于眼眶边界的位置来估算视线方向（Yaw/Pitch角度）
"""

import numpy as np
from typing import Dict, Optional, Any, List
from .face_mesh_indices import FaceMeshIndices


class GazeEstimator:
    """
    视线估算器

    通过计算虹膜中心相对于眼眶边界的位置，将其映射为视线角度（Yaw/Pitch）。
    """

    def __init__(self, max_yaw: float = 35.0, max_pitch: float = 25.0):
        """
        初始化视线估算器

        Args:
            max_yaw: 眼球最大水平转动角度（度），默认±35°
            max_pitch: 眼球最大垂直转动角度（度），默认±25°
        """
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.indices = FaceMeshIndices

    def estimate_gaze(self, landmarks: List[Any]) -> Optional[Dict]:
        """
        从面部关键点估算视线方向

        Args:
            landmarks: MediaPipe Face Mesh 检测到的关键点列表
                       每个landmark包含 x, y, z 属性（归一化坐标）

        Returns:
            包含视线信息的字典，如果检测失败返回 None
            {
                "left_eye": {"yaw": float, "pitch": float},
                "right_eye": {"yaw": float, "pitch": float},
                "averaged": {"yaw": float, "pitch": float},
                "iris_3d": {
                    "left": {"x": float, "y": float, "z": float},
                    "right": {"x": float, "y": float, "z": float}
                }
            }
        """
        if landmarks is None or len(landmarks) < 478:
            return None

        try:
            # 计算左眼视线
            left_gaze = self._compute_eye_gaze(
                landmarks,
                iris_center_idx=self.indices.LEFT_IRIS_CENTER,
                eye_bounds=self.indices.get_left_eye_bounds()
            )

            # 计算右眼视线
            right_gaze = self._compute_eye_gaze(
                landmarks,
                iris_center_idx=self.indices.RIGHT_IRIS_CENTER,
                eye_bounds=self.indices.get_right_eye_bounds()
            )

            if left_gaze is None or right_gaze is None:
                return None

            # 计算双眼平均
            avg_yaw = (left_gaze['yaw'] + right_gaze['yaw']) / 2
            avg_pitch = (left_gaze['pitch'] + right_gaze['pitch']) / 2

            # 获取虹膜3D坐标
            left_iris = landmarks[self.indices.LEFT_IRIS_CENTER]
            right_iris = landmarks[self.indices.RIGHT_IRIS_CENTER]

            return {
                "left_eye": {
                    "yaw": round(left_gaze['yaw'], 2),
                    "pitch": round(left_gaze['pitch'], 2)
                },
                "right_eye": {
                    "yaw": round(right_gaze['yaw'], 2),
                    "pitch": round(right_gaze['pitch'], 2)
                },
                "averaged": {
                    "yaw": round(avg_yaw, 2),
                    "pitch": round(avg_pitch, 2)
                },
                "iris_3d": {
                    "left": {
                        "x": round(left_iris.x, 4),
                        "y": round(left_iris.y, 4),
                        "z": round(left_iris.z, 4)
                    },
                    "right": {
                        "x": round(right_iris.x, 4),
                        "y": round(right_iris.y, 4),
                        "z": round(right_iris.z, 4)
                    }
                }
            }

        except (IndexError, AttributeError) as e:
            print(f"视线估算错误: {e}")
            return None

    def _compute_eye_gaze(
        self,
        landmarks: List[Any],
        iris_center_idx: int,
        eye_bounds: Dict[str, int]
    ) -> Optional[Dict[str, float]]:
        """
        计算单只眼睛的视线方向

        Args:
            landmarks: 面部关键点列表
            iris_center_idx: 虹膜中心索引
            eye_bounds: 眼眶边界索引字典 {'outer', 'inner', 'top', 'bottom'}

        Returns:
            {"yaw": float, "pitch": float} 或 None
        """
        try:
            # 获取虹膜中心坐标
            iris = landmarks[iris_center_idx]
            iris_x, iris_y = iris.x, iris.y

            # 获取眼眶边界坐标
            outer = landmarks[eye_bounds['outer']]
            inner = landmarks[eye_bounds['inner']]
            top = landmarks[eye_bounds['top']]
            bottom = landmarks[eye_bounds['bottom']]

            # 计算眼眶的水平和垂直范围
            eye_width = abs(inner.x - outer.x)
            eye_height = abs(bottom.y - top.y)

            if eye_width < 1e-6 or eye_height < 1e-6:
                return None

            # 计算虹膜在眼眶中的相对位置 (0-1)
            # 水平方向：outer -> inner
            eye_left = min(outer.x, inner.x)
            iris_x_ratio = (iris_x - eye_left) / eye_width

            # 垂直方向：top -> bottom
            eye_top = min(top.y, bottom.y)
            iris_y_ratio = (iris_y - eye_top) / eye_height

            # 将相对位置映射为角度
            # 中心位置 (0.5, 0.5) 对应 (0°, 0°)
            # 注意：左眼和右眼的水平方向需要考虑镜像
            yaw = (iris_x_ratio - 0.5) * 2 * self.max_yaw
            pitch = (iris_y_ratio - 0.5) * 2 * self.max_pitch

            # 反转pitch方向（向上看为正）
            pitch = -pitch

            return {"yaw": yaw, "pitch": pitch}

        except (IndexError, AttributeError, ZeroDivisionError):
            return None

    def get_iris_positions(self, landmarks: List[Any]) -> Optional[Dict]:
        """
        获取虹膜中心的像素坐标（用于可视化）

        Args:
            landmarks: 面部关键点列表

        Returns:
            {"left": (x, y), "right": (x, y)} 或 None
        """
        if landmarks is None or len(landmarks) < 478:
            return None

        try:
            left_iris = landmarks[self.indices.LEFT_IRIS_CENTER]
            right_iris = landmarks[self.indices.RIGHT_IRIS_CENTER]

            return {
                "left": (left_iris.x, left_iris.y),
                "right": (right_iris.x, right_iris.y)
            }
        except (IndexError, AttributeError):
            return None

    def get_eye_centers(self, landmarks: List[Any]) -> Optional[Dict]:
        """
        获取眼睛中心坐标（用于绘制视线箭头起点）

        Args:
            landmarks: 面部关键点列表

        Returns:
            {"left": (x, y), "right": (x, y)} 或 None
        """
        if landmarks is None or len(landmarks) < 478:
            return None

        try:
            # 左眼中心 = (外角 + 内角) / 2
            left_outer = landmarks[self.indices.LEFT_EYE_OUTER]
            left_inner = landmarks[self.indices.LEFT_EYE_INNER]
            left_center = (
                (left_outer.x + left_inner.x) / 2,
                (left_outer.y + left_inner.y) / 2
            )

            # 右眼中心
            right_outer = landmarks[self.indices.RIGHT_EYE_OUTER]
            right_inner = landmarks[self.indices.RIGHT_EYE_INNER]
            right_center = (
                (right_outer.x + right_inner.x) / 2,
                (right_outer.y + right_inner.y) / 2
            )

            return {
                "left": left_center,
                "right": right_center
            }
        except (IndexError, AttributeError):
            return None
