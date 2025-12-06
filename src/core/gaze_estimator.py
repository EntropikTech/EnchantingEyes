"""
视线估算器模块

基于3D向量法估算视线方向（Yaw/Pitch角度）和眼睑开合角度。

算法原理：
1. 通过内外眼角中点 + Z轴向后平移来估算眼球中心的3D坐标
2. 构建从眼球中心指向虹膜中心的3D向量
3. 根据该向量计算水平偏航角(Yaw)和垂直俯仰角(Pitch)
4. 计算眼球中心指向上/下眼睑的向量与垂直轴的夹角作为眼睑开合角度

坐标系说明（MediaPipe Face Mesh）：
- x: 归一化到 [0.0, 1.0]，向右为正
- y: 归一化到 [0.0, 1.0]，向下为正
- z: 以头部中心为原点，值越小表示离相机越近
"""

import numpy as np
from typing import Dict, Optional, Any, List
from .face_mesh_indices import FaceMeshIndices
from .constants import GazeConstants, EyelidConstants


class GazeEstimator:
    """
    视线估算器

    使用3D向量法计算视线方向和眼睑开合角度。
    """

    def __init__(
        self,
        max_yaw: float = None,
        max_pitch: float = None,
        eye_center_z_offset: float = None
    ):
        """
        初始化视线估算器

        Args:
            max_yaw: 眼球最大水平转动角度（度），默认使用 GazeConstants.MAX_YAW
            max_pitch: 眼球最大垂直转动角度（度），默认使用 GazeConstants.MAX_PITCH
            eye_center_z_offset: 眼球中心Z轴偏移量，默认使用 GazeConstants.EYE_CENTER_Z_OFFSET
        """
        self.max_yaw = max_yaw if max_yaw is not None else GazeConstants.MAX_YAW
        self.max_pitch = max_pitch if max_pitch is not None else GazeConstants.MAX_PITCH
        self.eye_center_z_offset = (
            eye_center_z_offset if eye_center_z_offset is not None
            else GazeConstants.EYE_CENTER_Z_OFFSET
        )
        self.indices = FaceMeshIndices

    def estimate_gaze(self, landmarks: List[Any]) -> Optional[Dict]:
        """
        从面部关键点估算视线方向和眼睑开合角度

        Args:
            landmarks: MediaPipe Face Mesh 检测到的关键点列表
                       每个landmark包含 x, y, z 属性（归一化坐标）

        Returns:
            包含视线和眼睑信息的字典，如果检测失败返回 None
            {
                "left_eye": {"yaw": float, "pitch": float},
                "right_eye": {"yaw": float, "pitch": float},
                "averaged": {"yaw": float, "pitch": float},
                "iris_3d": {
                    "left": {"x": float, "y": float, "z": float},
                    "right": {"x": float, "y": float, "z": float}
                },
                "eyelid": {
                    "left": {"upper": float, "lower": float},
                    "right": {"upper": float, "lower": float}
                },
                "eye_center_3d": {
                    "left": {"x": float, "y": float, "z": float},
                    "right": {"x": float, "y": float, "z": float}
                }
            }
        """
        if landmarks is None or len(landmarks) < 478:
            return None

        try:
            # 计算左眼视线（使用3D向量法）
            left_gaze = self._compute_eye_gaze(
                landmarks,
                iris_center_idx=self.indices.LEFT_IRIS_CENTER,
                inner_idx=self.indices.LEFT_EYE_INNER,
                outer_idx=self.indices.LEFT_EYE_OUTER
            )

            # 计算右眼视线
            right_gaze = self._compute_eye_gaze(
                landmarks,
                iris_center_idx=self.indices.RIGHT_IRIS_CENTER,
                inner_idx=self.indices.RIGHT_EYE_INNER,
                outer_idx=self.indices.RIGHT_EYE_OUTER
            )

            if left_gaze is None or right_gaze is None:
                return None

            # 计算左眼眼睑角度
            left_eyelid = self._compute_eyelid_angles(
                landmarks,
                eye_center=left_gaze['eye_center'],
                upper_idx=self.indices.LEFT_EYE_TOP,
                lower_idx=self.indices.LEFT_EYE_BOTTOM
            )

            # 计算右眼眼睑角度
            right_eyelid = self._compute_eyelid_angles(
                landmarks,
                eye_center=right_gaze['eye_center'],
                upper_idx=self.indices.RIGHT_EYE_TOP,
                lower_idx=self.indices.RIGHT_EYE_BOTTOM
            )

            # 计算双眼平均
            avg_yaw = (left_gaze['yaw'] + right_gaze['yaw']) / 2
            avg_pitch = (left_gaze['pitch'] + right_gaze['pitch']) / 2

            # 获取虹膜3D坐标
            left_iris = landmarks[self.indices.LEFT_IRIS_CENTER]
            right_iris = landmarks[self.indices.RIGHT_IRIS_CENTER]

            result = {
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
                },
                "eyelid": {
                    "left": {
                        "upper": round(left_eyelid['upper'], 2) if left_eyelid else None,
                        "lower": round(left_eyelid['lower'], 2) if left_eyelid else None
                    },
                    "right": {
                        "upper": round(right_eyelid['upper'], 2) if right_eyelid else None,
                        "lower": round(right_eyelid['lower'], 2) if right_eyelid else None
                    }
                },
                "eye_center_3d": {
                    "left": {
                        "x": round(left_gaze['eye_center'][0], 4),
                        "y": round(left_gaze['eye_center'][1], 4),
                        "z": round(left_gaze['eye_center'][2], 4)
                    },
                    "right": {
                        "x": round(right_gaze['eye_center'][0], 4),
                        "y": round(right_gaze['eye_center'][1], 4),
                        "z": round(right_gaze['eye_center'][2], 4)
                    }
                }
            }

            return result

        except (IndexError, AttributeError) as e:
            print(f"视线估算错误: {e}")
            return None

    def _compute_eye_center_3d(
        self,
        landmarks: List[Any],
        inner_idx: int,
        outer_idx: int
    ) -> np.ndarray:
        """
        计算眼球中心的3D坐标

        通过内眼角和外眼角连线的中点，沿Z轴向后平移来估算眼球中心。

        Args:
            landmarks: 面部关键点列表
            inner_idx: 内眼角索引
            outer_idx: 外眼角索引

        Returns:
            眼球中心3D坐标 np.array([x, y, z])
        """
        inner = landmarks[inner_idx]
        outer = landmarks[outer_idx]

        # 计算内外眼角中点
        center_x = (inner.x + outer.x) / 2
        center_y = (inner.y + outer.y) / 2
        center_z = (inner.z + outer.z) / 2

        # 沿Z轴向后（减小Z值，更靠近头部内部）平移
        center_z -= self.eye_center_z_offset

        return np.array([center_x, center_y, center_z])

    def _compute_eye_gaze(
        self,
        landmarks: List[Any],
        iris_center_idx: int,
        inner_idx: int,
        outer_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        使用3D向量法计算单只眼睛的视线方向

        构建从眼球中心指向虹膜中心的3D向量，计算Yaw/Pitch角度。

        Args:
            landmarks: 面部关键点列表
            iris_center_idx: 虹膜中心索引
            inner_idx: 内眼角索引
            outer_idx: 外眼角索引

        Returns:
            {"yaw": float, "pitch": float, "eye_center": np.ndarray} 或 None
        """
        try:
            # 1. 计算眼球中心3D坐标
            eye_center = self._compute_eye_center_3d(landmarks, inner_idx, outer_idx)

            # 2. 获取虹膜中心3D坐标
            iris = landmarks[iris_center_idx]
            iris_3d = np.array([iris.x, iris.y, iris.z])

            # 3. 构建从眼球中心指向虹膜中心的向量
            gaze_vector = iris_3d - eye_center

            # 4. 归一化向量
            norm = np.linalg.norm(gaze_vector)
            if norm < 1e-6:
                return None
            gaze_vector = gaze_vector / norm

            # 5. 计算Yaw角度（水平偏转）
            # Yaw = arctan2(x, -z)
            # 正值表示向右看（因为x向右为正）
            # 注意：MediaPipe的z值越小表示越靠近相机，所以需要取反
            yaw = np.degrees(np.arctan2(gaze_vector[0], -gaze_vector[2]))

            # 6. 计算Pitch角度（垂直偏转）
            # Pitch = arcsin(-y)
            # 正值表示向上看（因为y向下为正，需要取反）
            pitch = np.degrees(np.arcsin(-gaze_vector[1]))

            # 7. 限制在最大范围内
            yaw = np.clip(yaw, -self.max_yaw, self.max_yaw)
            pitch = np.clip(pitch, -self.max_pitch, self.max_pitch)

            return {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "eye_center": eye_center
            }

        except (IndexError, AttributeError, ZeroDivisionError):
            return None

    def _compute_eyelid_angles(
        self,
        landmarks: List[Any],
        eye_center: np.ndarray,
        upper_idx: int,
        lower_idx: int
    ) -> Optional[Dict[str, float]]:
        """
        计算眼睑开合角度

        通过计算眼球中心指向上/下眼睑中点的向量与垂直轴的夹角。

        角度定义（相对于眼球中心的垂直轴）：
        - 0度 = 垂直向上/向下（完全睁开）
        - 90度 = 水平（眼睑覆盖眼球中心）

        Args:
            landmarks: 面部关键点列表
            eye_center: 眼球中心3D坐标
            upper_idx: 上眼睑中点索引
            lower_idx: 下眼睑中点索引

        Returns:
            {"upper": float, "lower": float} 或 None
            - upper: 上眼睑角度，值越小表示眼睛睁得越大
            - lower: 下眼睑角度，值越小表示眼睛睁得越大
        """
        try:
            # 获取上下眼睑中点3D坐标
            upper = landmarks[upper_idx]
            lower = landmarks[lower_idx]
            upper_3d = np.array([upper.x, upper.y, upper.z])
            lower_3d = np.array([lower.x, lower.y, lower.z])

            # 构建从眼球中心指向眼睑的向量
            upper_vec = upper_3d - eye_center
            lower_vec = lower_3d - eye_center

            # 归一化
            upper_norm = np.linalg.norm(upper_vec)
            lower_norm = np.linalg.norm(lower_vec)
            if upper_norm < 1e-6 or lower_norm < 1e-6:
                return None
            upper_vec = upper_vec / upper_norm
            lower_vec = lower_vec / lower_norm

            # 计算上眼睑角度
            # 上眼睑在眼球上方，向上垂直轴为 (0, -1, 0)（因为y向下为正）
            # 使用 arccos 计算向量与垂直轴的夹角
            # upper_vec[1] 是 y 分量，向上时为负值
            # arccos(-y) 得到与向上垂直轴的夹角
            upper_angle = np.degrees(np.arccos(np.clip(-upper_vec[1], -1.0, 1.0)))

            # 计算下眼睑角度
            # 下眼睑在眼球下方，向下垂直轴为 (0, 1, 0)
            # lower_vec[1] 是 y 分量，向下时为正值
            lower_angle = np.degrees(np.arccos(np.clip(lower_vec[1], -1.0, 1.0)))

            # 限制在合理范围内
            upper_angle = np.clip(
                upper_angle,
                EyelidConstants.UPPER_EYELID_OPEN,
                EyelidConstants.UPPER_EYELID_CLOSED
            )
            lower_angle = np.clip(
                lower_angle,
                EyelidConstants.LOWER_EYELID_OPEN,
                EyelidConstants.LOWER_EYELID_CLOSED
            )

            return {
                "upper": float(upper_angle),
                "lower": float(lower_angle)
            }

        except (IndexError, AttributeError):
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
