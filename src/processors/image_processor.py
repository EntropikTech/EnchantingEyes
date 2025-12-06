"""
图片处理器模块

处理单张图片，提取视线数据
"""

import cv2
import json
import mediapipe as mp
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from ..core.gaze_estimator import GazeEstimator
from ..visualization.drawer import GazeDrawer


class ImageProcessor:
    """图片处理器"""

    def __init__(
        self,
        gaze_estimator: Optional[GazeEstimator] = None,
        viz_level: int = 1
    ):
        """
        初始化图片处理器

        Args:
            gaze_estimator: 视线估算器实例，如果为None则创建默认实例
            viz_level: 可视化级别 (1=简单, 2=标准, 3=完整网格)
        """
        self.gaze_estimator = gaze_estimator or GazeEstimator()
        self.drawer = GazeDrawer(viz_level=viz_level)

        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,   # 静态图片模式
            max_num_faces=1,
            refine_landmarks=True,    # 关键：启用虹膜检测
            min_detection_confidence=0.5
        )

    def process_image(
        self,
        input_path: str,
        output_json_path: Optional[str] = None,
        output_image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理单张图片

        Args:
            input_path: 输入图片路径
            output_json_path: 输出JSON文件路径（可选）
            output_image_path: 输出可视化图片路径（可选）

        Returns:
            处理结果字典
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {input_path}")

        # 读取图片
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"无法读取图片文件: {input_path}")

        height, width = image.shape[:2]

        # 转换颜色空间 BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 运行 Face Mesh 检测
        results = self.face_mesh.process(rgb_image)

        # 准备结果数据
        result = {
            "image_info": {
                "path": str(input_path),
                "resolution": [width, height]
            },
            "face_detected": False,
            "gaze": None,
            "iris_3d": None,
            "left_eye": None,
            "right_eye": None,
            "eyelid": None
        }

        annotated_image = image.copy()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 估算视线
            gaze_result = self.gaze_estimator.estimate_gaze(landmarks)

            if gaze_result:
                result["face_detected"] = True
                result["gaze"] = gaze_result["averaged"]
                result["left_eye"] = gaze_result["left_eye"]
                result["right_eye"] = gaze_result["right_eye"]
                result["iris_3d"] = gaze_result["iris_3d"]
                result["eyelid"] = gaze_result.get("eyelid")

                # 绘制可视化
                annotated_image = self.drawer.draw_gaze(
                    annotated_image, landmarks, gaze_result, (width, height)
                )

        # 保存JSON
        if output_json_path:
            output_json_path = Path(output_json_path)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        # 保存可视化图片
        if output_image_path:
            output_image_path = Path(output_image_path)
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image_path), annotated_image)

        # 添加标注图片到结果中（用于显示）
        result["annotated_image"] = annotated_image

        return result

    def show_result(self, result: Dict[str, Any], window_name: str = "Gaze Detection"):
        """
        显示处理结果

        Args:
            result: process_image 返回的结果字典
            window_name: 窗口名称
        """
        if "annotated_image" not in result:
            print("无可显示的图片")
            return

        image = result["annotated_image"]
        cv2.imshow(window_name, image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __del__(self):
        """释放资源"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
