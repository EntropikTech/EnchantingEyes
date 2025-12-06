import argparse 


def parse_args():
    """
    解析命令行参数。

    Args:
        无

    Returns:
        argparse.Namespace: 包含解析后的命令行参数的命名空间对象。

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./output"
    )
    parser.add_argument(
        "--log_path", 
        type=str, 
        default="./log"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-5
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=7
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
