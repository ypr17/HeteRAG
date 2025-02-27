import json
import random
from sklearn.model_selection import train_test_split
from ark_nlp.dataset import TwinTowersSentenceClassificationDataset as Dataset
from ark_nlp.processor.tokenizer.transfomer import SentenceTokenizer as Tokenizer
import pandas as pd
import os
from tqdm import tqdm

# 定义chunking函数
def chunk_text(text, tokenizer, chunk_size=32):
    """
    将文本分割为固定大小的chunks（每个chunk包含chunk_size个tokens）。
    """
    tokens = tokenizer.tokenize(text)  # 使用tokenizer将文本分割为tokens
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]  # 获取当前chunk
        chunk_text = tokenizer.convert_tokens_to_string(chunk)  # 将tokens转换回文本
        chunks.append(chunk_text)
    return chunks

def get_chunk_with_context(tokenizer, chunks, index, title):
    """
    获取当前chunk及其上下文，并按照指令模板结构化
    """
    current_chunk = chunks[index]
    prev_chunk = chunks[index - 1] if index > 0 else ""
    next_chunk = chunks[index + 1] if index < len(chunks) - 1 else ""

    # 结构化指令模板（用[SEP]分隔不同部分）
    prompt_parts = []
    if prev_chunk:
        prompt_parts.append(f"Previous text: {prev_chunk}")
    prompt_parts.append(f"Current text: {current_chunk}")
    if next_chunk:
        prompt_parts.append(f"Next text: {next_chunk}")
    prompt_parts.append(f"Title: {title}")

    # 使用[SEP]连接所有部分，并在末尾添加[SEP]（模型会自动添加[CLS]）
    sep = tokenizer.sep_token  # 获取模型特定的分隔符（如 "[SEP]"）
    combined_text = f" {sep} ".join(prompt_parts) + f" {sep}"

    return combined_text.strip()

def load_data(data_path, tokenizer):
    # 读取corpus.jsonl
    corpus = {}
    corpus_title = {}
    with open(os.path.join(data_path, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus[data["_id"]] = data["text"]
            corpus_title[data["_id"]] = data["title"]

    # 读取queries.jsonl
    queries = {}
    with open(os.path.join(data_path, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries[data["_id"]] = data["text"]
            
    # 读取qrels/train.jsonl并提取正样本
    if 'nfcorpus' in data_path:
        split = 'dev'
    else:
        split = 'train'
    print("########## Using " + split + " ##########")
    positive_samples = []
    with open(os.path.join(data_path, "qrels", split+".jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["score"] == "1" or data["score"] == "2":  # 只保留正样本  #####################################不同data
                positive_samples.append((data["query-id"], data["corpus-id"]))
    
    # 随机选取等量的负样本
    negative_samples = []
    corpus_ids = list(corpus.keys())
    for query_id, pos_corpus_id in positive_samples:
        while True:
            neg_corpus_id = random.choice(corpus_ids)
            if neg_corpus_id != pos_corpus_id:  # 确保负样本的corpus-id与正样本不同
                negative_samples.append((query_id, neg_corpus_id))
                break
            
    # 合并正样本和负样本，并对text2进行chunking
    samples = []
    for query_id, corpus_id in tqdm(positive_samples):
        text2_chunks = chunk_text(corpus[corpus_id], tokenizer=tokenizer, chunk_size=32)  # 对text2进行chunking
        for idx in range(len(text2_chunks)):
            combined_text = get_chunk_with_context(tokenizer, text2_chunks, idx, corpus_title[corpus_id])  # 获取带上下文的chunk
            samples.append({
                "id": f"{query_id}_{corpus_id}_{idx}",  # 添加chunk索引以区分不同chunk
                "text1": queries[query_id],
                "text2": combined_text,  # 使用带上下文的chunk
                "label": 1  # 正样本
            })
    for query_id, corpus_id in tqdm(negative_samples):
        text2_chunks = chunk_text(corpus[corpus_id], tokenizer=tokenizer, chunk_size=32)  # 对text2进行chunking
        for idx in range(len(text2_chunks)):
            combined_text = get_chunk_with_context(tokenizer, text2_chunks, idx, corpus_title[corpus_id])  # 获取带上下文的chunk
            samples.append({
                "id": f"{query_id}_{corpus_id}_{idx}",  # 添加chunk索引以区分不同chunk
                "text1": queries[query_id],
                "text2": combined_text,  # 使用带上下文的chunk
                "label": 0  # 负样本
            })
            
    # 打乱样本顺序
    random.shuffle(samples)
    print(samples[0])
    
    # if 'nfcorpus' in data_path:
    #     samples = samples[:50000]

    # 转换为DataFrame
    df = pd.DataFrame(samples)

    # 按照9:1划分train和dev
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42)

    # 打印划分后的数据集大小
    print(f"Train set size: {len(train_df)}")
    print(f"Dev set size: {len(dev_df)}")
    
    train_data_df = (train_df.rename(columns={'text1': 'text_a', 'text2': 'text_b'})
                 .loc[:,['text_a', 'text_b', 'label']])
    dev_data_df = (dev_df.rename(columns={'text1': 'text_a', 'text2': 'text_b'})
                    .loc[:,['text_a', 'text_b', 'label']])
    cosent_train_dataset = Dataset(train_data_df)
    cosent_dev_dataset = Dataset(dev_data_df)
    
    return cosent_train_dataset, cosent_dev_dataset, split