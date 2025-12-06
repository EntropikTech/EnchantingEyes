"""
MediaPipe Face Mesh 关键点索引定义

MediaPipe Face Mesh 在启用 refine_landmarks=True 时提供 478 个关键点：
- 468 个面部网格点
- 10 个虹膜关键点（左眼 468-472，右眼 473-477）
"""


class FaceMeshIndices:
    """Face Mesh 关键点索引常量"""

    # ===== 虹膜关键点 =====
    # 左眼虹膜 (5个点)
    LEFT_IRIS_CENTER = 468  # 左眼虹膜中心
    LEFT_IRIS = [468, 469, 470, 471, 472]

    # 右眼虹膜 (5个点)
    RIGHT_IRIS_CENTER = 473  # 右眼虹膜中心
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # ===== 眼眶边界关键点 =====
    # 用于计算虹膜相对于眼眶的位置

    # 左眼眶边界
    LEFT_EYE_OUTER = 33   # 左眼外角
    LEFT_EYE_INNER = 133  # 左眼内角
    LEFT_EYE_TOP = 159    # 左眼上缘
    LEFT_EYE_BOTTOM = 145  # 左眼下缘

    # 右眼眶边界
    RIGHT_EYE_OUTER = 362  # 右眼外角
    RIGHT_EYE_INNER = 263  # 右眼内角
    RIGHT_EYE_TOP = 386    # 右眼上缘
    RIGHT_EYE_BOTTOM = 374  # 右眼下缘

    # ===== 眼眶轮廓 (用于可视化) =====
    LEFT_EYE_CONTOUR = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        173, 157, 158, 159, 160, 161, 246
    ]

    RIGHT_EYE_CONTOUR = [
        362, 382, 381, 380, 374, 373, 390, 249, 263,
        466, 388, 387, 386, 385, 384, 398
    ]

    # ===== 面部姿态估算关键点 (用于头部姿态计算) =====
    NOSE_TIP = 1           # 鼻尖
    CHIN = 199             # 下巴
    LEFT_EYE_LEFT_CORNER = 33   # 左眼左角
    RIGHT_EYE_RIGHT_CORNER = 263  # 右眼右角
    LEFT_MOUTH_CORNER = 61   # 左嘴角
    RIGHT_MOUTH_CORNER = 291  # 右嘴角

    # 用于头部姿态估算的6个关键点
    POSE_LANDMARKS = [
        NOSE_TIP,
        CHIN,
        LEFT_EYE_LEFT_CORNER,
        RIGHT_EYE_RIGHT_CORNER,
        LEFT_MOUTH_CORNER,
        RIGHT_MOUTH_CORNER
    ]

    # ===== 面部区域索引 (用于 Level 3 分区着色) =====

    # 左眼区域（扩展）
    LEFT_EYE_REGION = [
        7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173,
        246, 130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30, 247
    ]

    # 右眼区域（扩展）
    RIGHT_EYE_REGION = [
        249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390,
        398, 466, 359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258,
        257, 259, 260, 467
    ]

    # 左眉毛区域
    LEFT_EYEBROW_REGION = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 156]

    # 右眉毛区域
    RIGHT_EYEBROW_REGION = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 383]

    # 虹膜区域
    LEFT_IRIS_REGION = [468, 469, 470, 471, 472]
    RIGHT_IRIS_REGION = [473, 474, 475, 476, 477]

    # 鼻子区域
    NOSE_REGION = [
        1, 2, 3, 4, 5, 6, 19, 44, 45, 48, 49, 51, 59, 60, 64, 75, 79, 94, 97,
        98, 99, 114, 115, 122, 125, 128, 129, 131, 134, 168, 188, 195, 196,
        197, 198, 209, 217, 236, 239, 240, 241, 242, 244, 245, 248, 250, 274,
        275, 278, 279, 281, 289, 290, 294, 305, 309, 326, 327, 328, 344, 351,
        354, 355, 357, 360, 392, 412, 419, 420, 437, 438, 439, 440
    ]

    # 嘴巴/嘴唇区域
    MOUTH_REGION = [
        0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 57, 61, 62, 72,
        73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96,
        146, 178, 179, 180, 181, 182, 183, 184, 185, 191, 267, 268, 269, 270,
        271, 272, 287, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311, 312,
        314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404,
        405, 406, 407, 408, 409, 415
    ]

    # 左脸颊区域
    LEFT_CHEEK_REGION = [
        36, 47, 50, 100, 101, 102, 104, 105, 106, 108, 109, 110, 111, 116, 117,
        118, 119, 120, 121, 123, 126, 137, 138, 139, 140, 147, 148, 149, 150,
        164, 165, 166, 167, 169, 170, 171, 172, 175, 176, 177, 186, 187, 192,
        202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215
    ]

    # 右脸颊区域
    RIGHT_CHEEK_REGION = [
        266, 277, 280, 329, 330, 331, 333, 334, 335, 337, 338, 339, 340, 345,
        346, 347, 348, 349, 350, 352, 355, 366, 367, 368, 369, 376, 377, 378,
        379, 391, 393, 394, 395, 396, 397, 400, 401, 410, 411, 416, 417, 418,
        421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433
    ]

    # 额头区域
    FOREHEAD_REGION = [
        10, 21, 54, 58, 67, 68, 69, 71, 103, 104, 107, 108, 109, 151, 162, 193,
        216, 251, 284, 288, 297, 298, 299, 301, 332, 333, 336, 337, 338, 383, 413
    ]

    # 下巴区域
    CHIN_REGION = [
        135, 136, 138, 140, 150, 152, 169, 170, 171, 172, 175, 176, 177, 194,
        199, 200, 201, 202, 204, 206, 207, 208, 210, 211, 212, 214, 364, 365,
        367, 369, 377, 378, 379, 394, 395, 396, 397, 400, 401, 418, 419, 420,
        421, 422, 424, 426, 427, 428, 430, 431, 432
    ]

    # 区域颜色映射 (BGR格式)
    REGION_COLORS = {
        'left_eye': (0, 255, 0),        # 绿色
        'right_eye': (0, 255, 0),       # 绿色
        'left_iris': (255, 255, 0),     # 青色
        'right_iris': (255, 255, 0),    # 青色
        'left_eyebrow': (255, 0, 255),  # 紫色
        'right_eyebrow': (255, 0, 255), # 紫色
        'nose': (0, 165, 255),          # 橙色
        'mouth': (0, 0, 255),           # 红色
        'left_cheek': (203, 192, 255),  # 粉色
        'right_cheek': (203, 192, 255), # 粉色
        'forehead': (200, 255, 255),    # 浅黄色
        'chin': (200, 200, 200),        # 灰色
        'default': (180, 180, 180),     # 默认灰色
    }

    @classmethod
    def get_all_regions(cls):
        """获取所有面部区域的索引映射"""
        return {
            'left_eye': cls.LEFT_EYE_REGION,
            'right_eye': cls.RIGHT_EYE_REGION,
            'left_iris': cls.LEFT_IRIS_REGION,
            'right_iris': cls.RIGHT_IRIS_REGION,
            'left_eyebrow': cls.LEFT_EYEBROW_REGION,
            'right_eyebrow': cls.RIGHT_EYEBROW_REGION,
            'nose': cls.NOSE_REGION,
            'mouth': cls.MOUTH_REGION,
            'left_cheek': cls.LEFT_CHEEK_REGION,
            'right_cheek': cls.RIGHT_CHEEK_REGION,
            'forehead': cls.FOREHEAD_REGION,
            'chin': cls.CHIN_REGION,
        }

    @classmethod
    def get_left_eye_bounds(cls):
        """获取左眼边界索引"""
        return {
            'outer': cls.LEFT_EYE_OUTER,
            'inner': cls.LEFT_EYE_INNER,
            'top': cls.LEFT_EYE_TOP,
            'bottom': cls.LEFT_EYE_BOTTOM
        }

    @classmethod
    def get_right_eye_bounds(cls):
        """获取右眼边界索引"""
        return {
            'outer': cls.RIGHT_EYE_OUTER,
            'inner': cls.RIGHT_EYE_INNER,
            'top': cls.RIGHT_EYE_TOP,
            'bottom': cls.RIGHT_EYE_BOTTOM
        }
