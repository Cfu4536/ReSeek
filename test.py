import pandas as pd
import json
import os


def print_row_details(row):
    """格式化打印单行数据的所有字段"""
    for column in row.index:
        print(f"【字段名】: {column}")
        value = row[column]

        # 对于字典或列表，使用 JSON 格式化以便于阅读
        if isinstance(value, (dict, list)):
            try:
                # ensure_ascii=False 保证中文正常显示，indent=4 提供良好的缩进
                pretty_value = json.dumps(value, indent=4, ensure_ascii=False)
                print(pretty_value)
            except TypeError:
                print(value)
        else:
            print(value)

        print("-" * 50)


def main():
    # 这里填写你上一步脚本生成的 Parquet 文件路径
    # 默认路径是 "./data/ReSeek_processed_direct_v1/train.parquet"
    file_path = "/opt/exps/ReSeek/data/ReSeek_processed_direct_v1/test.parquet"
    # file_path = "/opt/datasets/TencentBAC/ReSeek_train_test/test.parquet"
    # file_path = "/opt/exps/ReSeek/data/ReSeek_processed_direct_v1/train.parquet"

    if not os.path.exists(file_path):
        print(f"❌ 错误：未找到文件 {file_path}。")
        print("请检查你的路径是否正确，或者上一步的数据处理脚本是否成功运行。")
        return

    try:
        # 1. 读取 Parquet 文件
        df = pd.read_parquet(file_path)

        # 2. 打印基本信息
        print("✅ 数据读取成功！\n")
        print("=" * 50)
        print(f"数据集总行数: {len(df)}")
        print(f"包含的列名: {list(df.columns)}")
        print("=" * 50 + "\n")

        # 3. 打印第一行的详细结构
        print("👇 抽取第一行数据进行详细展示：\n")
        first_row = df.iloc[0]
        print_row_details(first_row)

    except Exception as e:
        print(f"读取数据时发生错误: {e}")


if __name__ == "__main__":
    main()