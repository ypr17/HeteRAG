import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


import torch.nn.functional as F
from ark_nlp.nn import BertConfig as ModuleConfig
from ark_nlp.nn import Bert
from ark_nlp.processor.tokenizer.transfomer import SentenceTokenizer as Tokenizer


from transformers import AutoTokenizer
import random

from modeling import CoSENT
from modeling import CoSENTTask
from data_hie import *

# 设置运行次数
num_epoches = 1
batch_size = 16
random.seed(42)

import argparse



def main():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # 添加参数
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    print(f"Model Path: {args.model_path}")
    print(f"Dataset Path: {args.dataset_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
    cosent_train_dataset, cosent_dev_dataset, split = load_data(args.dataset_path, tokenizer)
    tokenizer = Tokenizer(vocab=args.model_path, max_seq_len=512)
    cosent_train_dataset.convert_to_ids(tokenizer)
    cosent_dev_dataset.convert_to_ids(tokenizer)
    
    if 'jina' in args.model_path:
        from jina_bert_implementation.configuration_bert import JinaBertConfig
        bert_config = JinaBertConfig.from_pretrained(
            args.model_path,
            num_labels=len(cosent_train_dataset.cat2id)
        )
    else:
        from transformers import BertConfig
        bert_config = BertConfig.from_pretrained(
            args.model_path,
            num_labels=len(cosent_train_dataset.cat2id)
        )
        
    if 'bge' in args.model_path:
        dl_module = CoSENT(bert_config, model_path=args.model_path, pooling='cls')
    else:
        dl_module = CoSENT(bert_config, model_path=args.model_path, pooling='last_avg')
    
    dl_module = dl_module.cuda()
    
    param_optimizer = list(dl_module.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]     
    
    model = CoSENTTask(dl_module, 'adamw', None, cuda_device=0)
    
    
    model.fit(
        cosent_train_dataset,
        cosent_dev_dataset,
        lr=2e-5,
        epochs=num_epoches,
        batch_size=batch_size,
        params=optimizer_grouped_parameters
    )
    
    save_dir = '/data/hf-models/{}-{}-{}-hie-ins-ft-1-ep/'.format(args.dataset_path.strip('/').split('/')[-1], split, args.model_path.strip('/').split('/')[-1])
    model.module.bert.save_pretrained(save_dir)
    print(save_dir)
    
    model.fit(
        cosent_train_dataset,
        cosent_dev_dataset,
        lr=2e-5,
        epochs=num_epoches,
        batch_size=batch_size,
        params=optimizer_grouped_parameters
    )
    
    save_dir = '/data/hf-models/{}-{}-{}-hie-ins-ft-2-ep/'.format(args.dataset_path.strip('/').split('/')[-1], split, args.model_path.strip('/').split('/')[-1])
    model.module.bert.save_pretrained(save_dir)
    print(save_dir)


        

if __name__ == "__main__":
    main()