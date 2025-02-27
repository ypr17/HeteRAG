import click
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer

from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.wrappers import load_model
from jina_bert_implementation.modeling_bert import JinaBertModel
from FlagEmbedding import FlagModel

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

DEFAULT_CHUNKING_STRATEGY = 'fixed'
DEFAULT_CHUNK_SIZE = 32
DEFAULT_N_SENTENCES = 5
BATCH_SIZE = 1

@click.command()
@click.option(
    '--model-name',
    default='/share/yangxinhao-local/hf_models/jina-embeddings-v2-small-en',
    help='The name of the model to use.',
)
@click.option(
    '--strategy',
    default=DEFAULT_CHUNKING_STRATEGY,
    help='The chunking strategy to be applied.',
)
@click.option(
    '--chunk-size',
    default=DEFAULT_CHUNK_SIZE,
    help='The chunking size.',
)
@click.option(
    '--data-folder',
    default="/share/yangxinhao-local/data/scifact",
    help='The chunking size.',
)
@click.option(
    '--task-name', default='SciFactChunked', help='The evaluation task to perform.' 
    # SciFactChunked NarrativeQAChunked QuoraChunked FiQA2018Chunked TRECCOVIDChunked NFCorpusChunked
)
@click.option(
    '--eval-split', default='test', help='The name of the evaluation split in the task.'
)
@click.option(
    '--pooling-alg', default='naive-chunking', help='The pooling algorithm used, naive-chunking, late-chunking or hie-chunking.'
)
def main(model_name, strategy, chunk_size, task_name, eval_split, data_folder, pooling_alg):
    try:
        task_cls = globals()[task_name]
    except:
        raise ValueError(f'Unknown task name: {task_name}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if "jina" in model_name:
        model = JinaBertModel.from_pretrained(model_name).cuda()
    elif "bge" in model_name or "MedEmbed" in model_name:
        print("cls")
        model = FlagModel(model_name, pooling_method='cls')
    else: # e5
        model = FlagModel(model_name, pooling_method='mean')

    
    has_instructions = False

    chunking_args = {
        'chunk_size': chunk_size,
        'n_sentences': DEFAULT_N_SENTENCES,
        'chunking_strategy': strategy,
        'model_has_instructions': has_instructions,
    }

    # if torch.cuda.is_available():
    #     model = model.cuda()

    if "jina" in model_name:
        model.eval()
    else:
        model.model.eval()
    # model.eval()

    tasks = [
        task_cls(
            pooling_alg=pooling_alg,
            tokenizer=tokenizer,
            model_name=model_name,
            prune_size=None,
            **chunking_args,
        )
    ]

    evaluation = MTEB(
        tasks=tasks,
        pooling_alg=pooling_alg,
        tokenizer=tokenizer,
        prune_size=None,
        **chunking_args,
    )
    evaluation.run(
        model,
        tokenizer,
        output_folder='/data/tmps/hie-chunking-results/',
        eval_splits=[eval_split],
        overwrite_results=True,
        batch_size=BATCH_SIZE,
        encode_kwargs={'batch_size': BATCH_SIZE},
        data_folder=data_folder
    )
    print()
    print()


if __name__ == '__main__':
    main()
