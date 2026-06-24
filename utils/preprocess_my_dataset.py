import json
import os
from pprint import pprint

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def retrieval_ctxs(query):
    url = "http://localhost:8100/retrieve"

    payload = {
        "queries": [query],
        "topk": 10,
        "return_scores": True
    }

    res = requests.post(url, json=payload).json()
    result = res['result'][0]
    documents = [{"text": d['document']['contents'], "score": d['score'], "id": d['document']['id']} for d in result]
    return documents


def process_dataset(test_name, output_path):
    file_path = "/opt/datasets/TencentBAC/ReSeek_train_test/test.parquet"
    data_list = []

    # 1. 读取 Parquet 文件
    df = pd.read_parquet(file_path)
    df = df[df['data_source'] == test_name]

    for i in tqdm(range(len(df))):
        row = df.iloc[i]

        extra_info = row['extra_info']
        question = extra_info.get('question', '')

        # 3. 提取 Answers
        # reward_model -> ground_truth -> target 里面是一个 numpy array
        reward_model = row['reward_model']
        target_array = reward_model.get('ground_truth', {}).get('target', np.array([]))

        # 将 numpy array 转成标准的 Python list，以便 JSON 序列化
        answers = target_array.tolist() if isinstance(target_array, np.ndarray) else list(target_array)

        # 4. 调用检索并组装 ctxs
        ctxs = retrieval_ctxs(question)

        # 5. 组装最终单个 item
        item = {
            "question": question,
            "answers": answers,
            "qa_pairs": None,
            "ctxs": ctxs
        }
        # pprint(item)
        data_list.append(item)

    # 6. 写入文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def main():
    process_dataset(test_name="nq", output_path="/opt/exps/RankCoT/data/ReSeek_dataset/nq/test.json")


if __name__ == '__main__':
    main()
