"""
EnchantingEyes - 视线估算工具

基于 MediaPipe Face Mesh 的虹膜检测，估算视线方向（Yaw/Pitch角度）
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 定义输出目录
OUTPUT_DIR = PROJECT_ROOT / "results"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="EnchantingEyes - 视线估算工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理视频并输出JSON结果
  python src/main.py --input video.mp4

  # 处理视频并生成可视化视频
  python src/main.py --input video.mp4 --output_video
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/videos/test_2.mp4",
        help="输入视频文件路径"
    )

    parser.add_argument(
        "--output_video", "-v",
        action="store_true",
        default=True,
        help="是否输出可视化视频"
    )

    parser.add_argument(
        "--max_yaw",
        type=float,
        default=35.0,
        help="眼球最大水平转动角度（度），默认35"
    )

    parser.add_argument(
        "--max_pitch",
        type=float,
        default=25.0,
        help="眼球最大垂直转动角度（度），默认25"
    )

    parser.add_argument(
        "--eye_center_z_offset",
        type=float,
        default=0.03,
        help="眼球中心Z轴向后偏移量（归一化坐标），默认0.03"
    )

    parser.add_argument(
        "--viz_level",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="可视化级别: 1=简单(虹膜+视线), 2=标准(+眼眶轮廓), 3=完整(478点分区网格)"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 检查输入文件是否存在
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 生成输出文件路径
    stem = input_path.stem
    output_json = OUTPUT_DIR / f"{stem}_gaze.json"
    output_video = OUTPUT_DIR / f"{stem}_output.mp4" if args.output_video else None

    print("=" * 50)
    print("EnchantingEyes - 视线估算工具")
    print("=" * 50)
    print(f"输入文件: {input_path}")
    print(f"输出JSON: {output_json}")
    if output_video:
        print(f"输出视频: {output_video}")
    print(f"角度限制: max_yaw={args.max_yaw}°, max_pitch={args.max_pitch}°")
    print(f"眼球中心Z偏移: {args.eye_center_z_offset}")
    print(f"可视化级别: {args.viz_level}")
    print("=" * 50)

    # 延迟导入（避免未安装依赖时启动报错）
    from src.core.gaze_estimator import GazeEstimator
    from src.processors.video_processor import VideoProcessor

    # 创建处理器
    gaze_estimator = GazeEstimator(
        max_yaw=args.max_yaw,
        max_pitch=args.max_pitch,
        eye_center_z_offset=args.eye_center_z_offset
    )
    processor = VideoProcessor(gaze_estimator, viz_level=args.viz_level)

    # 处理视频
    try:
        result = processor.process_video(
            input_path=str(input_path),
            output_json_path=str(output_json),
            output_video_path=str(output_video) if output_video else None
        )

        print("\n处理完成!")
        print(f"  总帧数: {result['total_frames']}")
        print(f"  检测帧数: {result['detected_frames']}")
        print(f"  检测率: {result['detection_rate']}%")
        print(f"  JSON输出: {result['output_json']}")
        if result['output_video']:
            print(f"  视频输出: {result['output_video']}")

    except Exception as e:
        print(f"处理错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
