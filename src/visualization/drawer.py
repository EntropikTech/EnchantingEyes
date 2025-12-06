"""
视线可视化绘制模块

绘制虹膜关键点、视线方向箭头和角度信息
支持三级可视化：简单、标准、完整网格
"""

import cv2
import numpy as np
import math
from enum import IntEnum
from typing import Dict, Any, List, Tuple, Optional

from ..core.face_mesh_indices import FaceMeshIndices


class VizLevel(IntEnum):
    """可视化级别"""
    SIMPLE = 1    # 简单：虹膜 + 视线箭头 + 文字
    STANDARD = 2  # 标准：Level 1 + 眼眶轮廓
    FULL = 3      # 完整：全部478点分区网格


class GazeDrawer:
    """视线可视化绘制器"""

    # 颜色定义 (BGR格式)
    COLOR_IRIS = (0, 255, 0)        # 绿色 - 虹膜
    COLOR_GAZE_ARROW = (0, 0, 255)  # 红色 - 视线箭头
    COLOR_EYE_CONTOUR = (255, 255, 0)  # 青色 - 眼眶轮廓
    COLOR_TEXT = (255, 255, 255)    # 白色 - 文字
    COLOR_TEXT_BG = (0, 0, 0)       # 黑色 - 文字背景

    def __init__(
        self,
        viz_level: int = 1,
        draw_iris: bool = True,
        draw_gaze_arrow: bool = True,
        draw_eye_contour: Optional[bool] = None,
        draw_text: bool = True,
        arrow_length: float = 50.0
    ):
        """
        初始化绘制器

        Args:
            viz_level: 可视化级别 (1=简单, 2=标准, 3=完整)
            draw_iris: 是否绘制虹膜中心点
            draw_gaze_arrow: 是否绘制视线方向箭头
            draw_eye_contour: 是否绘制眼眶轮廓（None时由viz_level控制）
            draw_text: 是否绘制角度文本
            arrow_length: 视线箭头长度（像素）
        """
        self.viz_level = viz_level
        self.draw_iris = draw_iris
        self.draw_gaze_arrow = draw_gaze_arrow
        # 根据 viz_level 自动设置眼眶轮廓绘制
        self.draw_eye_contour = draw_eye_contour if draw_eye_contour is not None else (viz_level >= 2)
        self.draw_text = draw_text
        self.draw_full_mesh = (viz_level >= 3)
        self.arrow_length = arrow_length

        self.indices = FaceMeshIndices

        # 预构建点到区域的映射表 (用于 Level 3)
        self._build_region_mapping()

    def _build_region_mapping(self):
        """构建特征点索引到区域名称的映射"""
        self.point_to_region = {}
        regions = self.indices.get_all_regions()
        for region_name, indices in regions.items():
            for idx in indices:
                self.point_to_region[idx] = region_name

    def draw_gaze(
        self,
        image: np.ndarray,
        landmarks: List[Any],
        gaze_result: Dict[str, Any],
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        在图像上绘制视线可视化

        Args:
            image: 输入图像 (BGR格式)
            landmarks: MediaPipe Face Mesh 关键点列表
            gaze_result: GazeEstimator.estimate_gaze() 的返回结果
            image_size: 图像尺寸 (width, height)

        Returns:
            绘制后的图像
        """
        if landmarks is None or gaze_result is None:
            return image

        width, height = image_size
        output = image.copy()

        # Level 3: 绘制完整面部网格（最底层，先绘制）
        if self.draw_full_mesh:
            self._draw_full_face_mesh(output, landmarks, width, height)

        # Level 2+: 绘制眼眶轮廓
        if self.draw_eye_contour:
            self._draw_eye_contours(output, landmarks, width, height)

        # Level 1+: 绘制虹膜中心点
        if self.draw_iris:
            self._draw_iris_points(output, landmarks, width, height)

        # Level 1+: 绘制视线箭头
        if self.draw_gaze_arrow:
            self._draw_gaze_arrows(output, landmarks, gaze_result, width, height)

        # Level 1+: 绘制角度文本和坐标信息
        if self.draw_text:
            self._draw_gaze_text(output, gaze_result, landmarks)

        return output

    def _draw_iris_points(
        self,
        image: np.ndarray,
        landmarks: List[Any],
        width: int,
        height: int
    ):
        """绘制虹膜中心点"""
        # 左眼虹膜
        left_iris = landmarks[self.indices.LEFT_IRIS_CENTER]
        left_x = int(left_iris.x * width)
        left_y = int(left_iris.y * height)
        cv2.circle(image, (left_x, left_y), 4, self.COLOR_IRIS, -1)
        cv2.circle(image, (left_x, left_y), 6, self.COLOR_IRIS, 1)

        # 右眼虹膜
        right_iris = landmarks[self.indices.RIGHT_IRIS_CENTER]
        right_x = int(right_iris.x * width)
        right_y = int(right_iris.y * height)
        cv2.circle(image, (right_x, right_y), 4, self.COLOR_IRIS, -1)
        cv2.circle(image, (right_x, right_y), 6, self.COLOR_IRIS, 1)

    def _draw_eye_contours(
        self,
        image: np.ndarray,
        landmarks: List[Any],
        width: int,
        height: int
    ):
        """绘制眼眶轮廓"""
        # 左眼轮廓
        left_points = []
        for idx in self.indices.LEFT_EYE_CONTOUR:
            pt = landmarks[idx]
            left_points.append((int(pt.x * width), int(pt.y * height)))
        if left_points:
            pts = np.array(left_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, self.COLOR_EYE_CONTOUR, 1)

        # 右眼轮廓
        right_points = []
        for idx in self.indices.RIGHT_EYE_CONTOUR:
            pt = landmarks[idx]
            right_points.append((int(pt.x * width), int(pt.y * height)))
        if right_points:
            pts = np.array(right_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, self.COLOR_EYE_CONTOUR, 1)

    def _draw_gaze_arrows(
        self,
        image: np.ndarray,
        landmarks: List[Any],
        gaze_result: Dict[str, Any],
        width: int,
        height: int
    ):
        """绘制视线方向箭头"""
        # 获取眼睛中心坐标
        left_outer = landmarks[self.indices.LEFT_EYE_OUTER]
        left_inner = landmarks[self.indices.LEFT_EYE_INNER]
        right_outer = landmarks[self.indices.RIGHT_EYE_OUTER]
        right_inner = landmarks[self.indices.RIGHT_EYE_INNER]

        # 左眼中心
        left_center_x = int((left_outer.x + left_inner.x) / 2 * width)
        left_center_y = int((left_outer.y + left_inner.y) / 2 * height)

        # 右眼中心
        right_center_x = int((right_outer.x + right_inner.x) / 2 * width)
        right_center_y = int((right_outer.y + right_inner.y) / 2 * height)

        # 获取视线角度
        left_yaw = gaze_result["left_eye"]["yaw"]
        left_pitch = gaze_result["left_eye"]["pitch"]
        right_yaw = gaze_result["right_eye"]["yaw"]
        right_pitch = gaze_result["right_eye"]["pitch"]

        # 绘制左眼视线箭头
        self._draw_arrow(
            image,
            (left_center_x, left_center_y),
            left_yaw,
            left_pitch
        )

        # 绘制右眼视线箭头
        self._draw_arrow(
            image,
            (right_center_x, right_center_y),
            right_yaw,
            right_pitch
        )

    def _draw_arrow(
        self,
        image: np.ndarray,
        start: Tuple[int, int],
        yaw: float,
        pitch: float
    ):
        """
        绘制单个视线箭头

        Args:
            image: 图像
            start: 箭头起点 (x, y)
            yaw: 水平偏转角（度），正值向右
            pitch: 垂直偏转角（度），正值向上
        """
        # 将角度转换为弧度
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)

        # 计算箭头终点
        # 注意：图像坐标系y轴向下，所以pitch要取负
        dx = self.arrow_length * math.sin(yaw_rad)
        dy = -self.arrow_length * math.sin(pitch_rad)

        end_x = int(start[0] + dx)
        end_y = int(start[1] + dy)

        # 绘制箭头
        cv2.arrowedLine(
            image,
            start,
            (end_x, end_y),
            self.COLOR_GAZE_ARROW,
            2,
            tipLength=0.3
        )

    def _format_coord(self, x: float, y: float, z: float) -> str:
        """格式化 3D 坐标为字符串"""
        return f"({x:.4f}, {y:.4f}, {z:.4f})"

    def _draw_gaze_text(
        self,
        image: np.ndarray,
        gaze_result: Dict[str, Any],
        landmarks: List[Any] = None
    ):
        """绘制视线角度、眼睑角度和坐标信息文本"""
        avg_gaze = gaze_result["averaged"]
        yaw = avg_gaze["yaw"]
        pitch = avg_gaze["pitch"]

        # 字体设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        line_height = 36
        padding = 5
        x, y = 10, 20

        # 收集所有要显示的文本行
        text_lines = []

        # 视线角度（第一行）
        text_lines.append(f"Yaw: {yaw:+.1f}  Pitch: {pitch:+.1f}")

        # 眼睑角度（第二行）
        if "eyelid" in gaze_result and gaze_result["eyelid"]:
            left_eyelid = gaze_result["eyelid"].get("left", {})
            right_eyelid = gaze_result["eyelid"].get("right", {})
            left_upper = left_eyelid.get("upper")
            right_upper = right_eyelid.get("upper")
            left_lower = left_eyelid.get("lower")
            right_lower = right_eyelid.get("lower")

            if all(v is not None for v in [left_upper, right_upper, left_lower, right_lower]):
                avg_upper = (left_upper + right_upper) / 2
                avg_lower = (left_lower + right_lower) / 2
                text_lines.append(f"Upper: {avg_upper:.1f}  Lower: {avg_lower:.1f}")

        # 坐标信息（需要 landmarks）
        if landmarks is not None:
            text_lines.append("")  # 空行分隔

            # 左眼坐标
            text_lines.append("=== Left Eye ===")
            left_iris = landmarks[self.indices.LEFT_IRIS_CENTER]
            text_lines.append(f"Iris:   {self._format_coord(left_iris.x, left_iris.y, left_iris.z)}")

            left_upper_pt = landmarks[self.indices.LEFT_EYE_TOP]
            text_lines.append(f"Upper:  {self._format_coord(left_upper_pt.x, left_upper_pt.y, left_upper_pt.z)}")

            left_lower_pt = landmarks[self.indices.LEFT_EYE_BOTTOM]
            text_lines.append(f"Lower:  {self._format_coord(left_lower_pt.x, left_lower_pt.y, left_lower_pt.z)}")

            left_outer_pt = landmarks[self.indices.LEFT_EYE_OUTER]
            text_lines.append(f"Outer:  {self._format_coord(left_outer_pt.x, left_outer_pt.y, left_outer_pt.z)}")

            left_inner_pt = landmarks[self.indices.LEFT_EYE_INNER]
            text_lines.append(f"Inner:  {self._format_coord(left_inner_pt.x, left_inner_pt.y, left_inner_pt.z)}")

            # 左眼球中心（从 gaze_result 获取）
            if "eye_center_3d" in gaze_result and gaze_result["eye_center_3d"]:
                left_center = gaze_result["eye_center_3d"].get("left", {})
                if left_center:
                    text_lines.append(f"Center: {self._format_coord(left_center['x'], left_center['y'], left_center['z'])}")

            text_lines.append("")  # 空行分隔

            # 右眼坐标
            text_lines.append("=== Right Eye ===")
            right_iris = landmarks[self.indices.RIGHT_IRIS_CENTER]
            text_lines.append(f"Iris:   {self._format_coord(right_iris.x, right_iris.y, right_iris.z)}")

            right_upper_pt = landmarks[self.indices.RIGHT_EYE_TOP]
            text_lines.append(f"Upper:  {self._format_coord(right_upper_pt.x, right_upper_pt.y, right_upper_pt.z)}")

            right_lower_pt = landmarks[self.indices.RIGHT_EYE_BOTTOM]
            text_lines.append(f"Lower:  {self._format_coord(right_lower_pt.x, right_lower_pt.y, right_lower_pt.z)}")

            right_outer_pt = landmarks[self.indices.RIGHT_EYE_OUTER]
            text_lines.append(f"Outer:  {self._format_coord(right_outer_pt.x, right_outer_pt.y, right_outer_pt.z)}")

            right_inner_pt = landmarks[self.indices.RIGHT_EYE_INNER]
            text_lines.append(f"Inner:  {self._format_coord(right_inner_pt.x, right_inner_pt.y, right_inner_pt.z)}")

            # 右眼球中心（从 gaze_result 获取）
            if "eye_center_3d" in gaze_result and gaze_result["eye_center_3d"]:
                right_center = gaze_result["eye_center_3d"].get("right", {})
                if right_center:
                    text_lines.append(f"Center: {self._format_coord(right_center['x'], right_center['y'], right_center['z'])}")

        # 计算背景矩形尺寸
        max_width = 0
        for line in text_lines:
            if line:  # 跳过空行
                (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_width = max(max_width, w)

        total_height = len(text_lines) * line_height

        # 绘制半透明背景矩形
        cv2.rectangle(
            image,
            (x - padding, y - line_height + padding),
            (x + max_width + padding, y + total_height - line_height + padding),
            self.COLOR_TEXT_BG,
            -1
        )

        # 绘制所有文本行
        current_y = y
        for line in text_lines:
            if line:  # 跳过空行只增加间距
                cv2.putText(
                    image,
                    line,
                    (x, current_y),
                    font,
                    font_scale,
                    self.COLOR_TEXT,
                    thickness
                )
            current_y += line_height

    def _draw_full_face_mesh(
        self,
        image: np.ndarray,
        landmarks: List[Any],
        width: int,
        height: int
    ):
        """
        绘制完整面部网格 (Level 3)

        使用 MediaPipe 的 FACEMESH_TESSELATION 连接关系绘制三角网格，
        并按区域着色。
        """
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        FACEMESH_TESSELATION = mp_face_mesh.FACEMESH_TESSELATION

        # 获取区域颜色
        region_colors = self.indices.REGION_COLORS
        default_color = region_colors.get('default', (180, 180, 180))

        # 绘制网格连接线
        for connection in FACEMESH_TESSELATION:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_pt = landmarks[start_idx]
                end_pt = landmarks[end_idx]

                start_x = int(start_pt.x * width)
                start_y = int(start_pt.y * height)
                end_x = int(end_pt.x * width)
                end_y = int(end_pt.y * height)

                # 使用起点的区域颜色
                region_name = self.point_to_region.get(start_idx, 'default')
                color = region_colors.get(region_name, default_color)

                # 降低颜色强度以达到半透明效果
                faded_color = tuple(int(c * 0.5) for c in color)

                # 绘制连接线
                cv2.line(image, (start_x, start_y), (end_x, end_y), faded_color, 1, cv2.LINE_AA)

        # 绘制所有特征点（按区域着色）
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # 确定该点所属区域
            region_name = self.point_to_region.get(idx, 'default')
            color = region_colors.get(region_name, default_color)

            # 绘制点（小圆点）
            cv2.circle(image, (x, y), 1, color, -1)
