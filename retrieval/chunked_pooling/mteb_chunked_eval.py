import logging
from typing import Any, Optional

import numpy as np
import torch

# from my_RetrievalEvaluator import my_RetrievalEvaluator
# import mteb
# mteb.evaluation.evaluators.RetrievalEvaluator = my_RetrievalEvaluator

from mteb.abstasks import AbsTask
from mteb.evaluation.evaluators import RetrievalEvaluator
from mteb.load_results.mteb_results import ScoresDict
from mteb.tasks import Retrieval
from tqdm import tqdm

from chunked_pooling import chunked_pooling
from chunked_pooling.chunking import Chunker

import pickle
import copy

logger = logging.getLogger(__name__)
                
def cos_sim(a, b):
    """Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """  # noqa: D402
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

class AbsTaskChunkedRetrieval(AbsTask):
    def __init__(
        self,
        chunking_strategy: str = None,
        pooling_alg: str = None,
        model_name: str = None,
        tokenizer: Optional[Any] = None,
        prune_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        n_sentences: Optional[int] = None,
        model_has_instructions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            self.retrieval_task = getattr(
                Retrieval,
                self.metadata_dict['dataset'].get('name', None)
                or self.metadata_dict.get('name'),
            )()
        except:
            logger.warning('Could not initialize retrieval_task')
        self.chunking_strategy = chunking_strategy
        self.chunker = Chunker(self.chunking_strategy)
        self.pooling_alg = pooling_alg
        self.tokenizer = tokenizer
        self.prune_size = prune_size
        self.model_has_instructions = model_has_instructions
        self.chunking_args = {
            'chunk_size': chunk_size,
            'n_sentences': n_sentences,
        }
        self.model_name = model_name
        
        self.long_late_chunking_embed_size = 512
        self.long_late_chunking_overlap_size = 256

    def load_data(self, **kwargs):
        self.retrieval_task.load_data(**kwargs)
        self.corpus = self.retrieval_task.corpus
        self.queries = self.retrieval_task.queries
        self.relevant_docs = self.retrieval_task.relevant_docs
        # prune dataset
        if self.prune_size:
            self.queries, self.corpus, self.relevant_docs = self._prune(
                self.queries, self.corpus, self.relevant_docs, self.prune_size
            )

    def calculate_metadata_metrics(self):
        self.retrieval_task.calculate_metadata_metrics()
        
    def _embed_with_overlap(self, model, model_inputs):
        len_tokens = len(model_inputs["input_ids"][0])

        if len_tokens > self.long_late_chunking_embed_size:
            indices = []
            for i in range(
                0,
                len_tokens,
                self.long_late_chunking_embed_size
                - self.long_late_chunking_overlap_size,
            ):
                start = i
                end = min(i + self.long_late_chunking_embed_size, len_tokens)
                indices.append((start, end))
        else:
            indices = [(0, len_tokens)]

        outputs = []
        for start, end in indices:
            batch_inputs = {k: v[:, start:end] for k, v in model_inputs.items()}

            with torch.no_grad():
                # model_output = model(**batch_inputs)
                if 'jina' in self.model_name:
                    model_output = model(**batch_inputs)
                else:
                    model_output = model.model(**batch_inputs)
                
            if start > 0:
                outputs.append(
                    model_output[0][:, self.long_late_chunking_overlap_size :]
                )
            else:
                outputs.append(model_output[0])

        return torch.cat(outputs, dim=1).to(model.device)

    def evaluate(
        self, model, tokenizer, split: str = "test", encode_kwargs: dict[str, Any] = {}, **kwargs
    ) -> dict[str, ScoresDict]:
        scores: dict[str, ScoresDict] = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )

            scores[hf_subset] = self._evaluate_monolingual(
                model,
                tokenizer,
                corpus,
                queries,
                relevant_docs,
                hf_subset,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )

        return scores

    def _evaluate_monolingual(
        self,
        model,
        tokenizer,
        corpus,
        queries,
        relevant_docs,
        lang=None,
        batch_size=1,
        encode_kwargs=None,
        **kwargs,
    ):
        # split corpus into chunks
        if self.pooling_alg == 'naive-chunking':
            corpus = self._apply_chunking(corpus, self.tokenizer)
            max_chunks = max([len(x) for x in corpus.values()])
            corpus = self._flatten_chunks(corpus)
                
            k_values = self._calculate_k_values(max_chunks)
            # determine the maximum number of documents to consider in a ranking
            max_k = int(max(k_values) / max_chunks)
            retriever = RetrievalEvaluator(
                model,
                k_values=k_values,
                encode_kwargs=(encode_kwargs or dict()),
                **kwargs,
            )
            results = retriever(corpus, queries)
            
            
        elif self.pooling_alg == 'hie-chunking':
            corpus_ids = list(corpus.keys())
            corpus_titles = {}
            for k in corpus_ids:
                corpus_titles[k] = corpus[k]['title']
            # corpus_titles = [corpus[k]['title'] for k in corpus_ids]
                
                
            corpus = self._apply_chunking(corpus, self.tokenizer)
            ori_corpus = copy.deepcopy(corpus)
            for corpus_id in corpus.keys():
                if len(corpus[corpus_id]) == 1:
                    # print(ori_corpus[corpus_id][0]['text'])
                    corpus[corpus_id][0]['text'] = ori_corpus[corpus_id][0]['text'] + corpus_titles[corpus_id]
                    continue
                for i in range(len(corpus[corpus_id])):
                    if i == 0:
                        corpus[corpus_id][i]['text'] =  ori_corpus[corpus_id][i]['text'] + ori_corpus[corpus_id][i+1]['text'] + corpus_titles[corpus_id]
                    elif i == len(corpus[corpus_id])-1:
                        corpus[corpus_id][i]['text'] =  ori_corpus[corpus_id][i-1]['text'] + ori_corpus[corpus_id][i]['text'] + corpus_titles[corpus_id]
                    else:
                        corpus[corpus_id][i]['text'] =  ori_corpus[corpus_id][i-1]['text'] + ori_corpus[corpus_id][i]['text'] + ori_corpus[corpus_id][i+1]['text'] + corpus_titles[corpus_id]

            max_chunks = max([len(x) for x in corpus.values()])
            corpus = self._flatten_chunks(corpus)
            k_values = self._calculate_k_values(max_chunks)
            # determine the maximum number of documents to consider in a ranking
            max_k = int(max(k_values) / max_chunks)
            
            retriever = RetrievalEvaluator(
                model,
                k_values=k_values,
                encode_kwargs=(encode_kwargs or dict()),
                **kwargs,
            )
            results = retriever(corpus, queries)
            
        elif self.pooling_alg == 'ft-hie-chunking':
            sep = tokenizer.sep_token  # 获取模型特定的分隔符（如 "[SEP]"）
            
            corpus_ids = list(corpus.keys())
            corpus_titles = {}
            for k in corpus_ids:
                corpus_titles[k] = corpus[k]['title']
            # corpus_titles = [corpus[k]['title'] for k in corpus_ids]
                
                
            corpus = self._apply_chunking(corpus, self.tokenizer)
            ori_corpus = copy.deepcopy(corpus) 
            for corpus_id in corpus.keys():
                for i in range(len(corpus[corpus_id])):
                    title = corpus_titles[corpus_id]
                    current_chunk = ori_corpus[corpus_id][i]['text']
                    prev_chunk = ori_corpus[corpus_id][i-1]['text'] if i > 0 else ""
                    next_chunk = ori_corpus[corpus_id][i+1]['text'] if i < len(corpus[corpus_id])-1 else ""
                    
                    # 结构化指令模板（用[SEP]分隔不同部分）
                    
                    prompt_parts = []
                    if prev_chunk:
                        prompt_parts.append(f"Previous text: {prev_chunk}")
                    prompt_parts.append(f"Current text: {current_chunk}")
                    if next_chunk:
                        prompt_parts.append(f"Next text: {next_chunk}")
                    prompt_parts.append(f"Title: {title}")
                    
                    # 使用[SEP]连接所有部分，并在末尾添加[SEP]（模型会自动添加[CLS]）
                    combined_text = f" {sep} ".join(prompt_parts) + f" {sep}"
                    combined_text = combined_text.strip()
                    
                    corpus[corpus_id][i]['text'] = combined_text
                    # print(corpus[corpus_id][i]['text'])
        
            max_chunks = max([len(x) for x in corpus.values()])
            corpus = self._flatten_chunks(corpus)
            k_values = self._calculate_k_values(max_chunks)
            # determine the maximum number of documents to consider in a ranking
            max_k = int(max(k_values) / max_chunks)
            
            retriever = RetrievalEvaluator(
                model,
                k_values=k_values,
                encode_kwargs=(encode_kwargs or dict()),
                **kwargs,
            )
            results = retriever(corpus, queries)
            
        elif self.pooling_alg == "new-hie-chunking": 
            corpus_ids = list(corpus.keys())
            corpus_titles = {}
            for k in corpus_ids:
                corpus_titles[k] = corpus[k]['title']
                
            corpus = self._apply_chunking(corpus, self.tokenizer)
            # ori_corpus = copy.deepcopy(corpus)
            for corpus_id in corpus.keys():
                for i in range(len(corpus[corpus_id])):
                    corpus[corpus_id][i]['title'] = corpus_titles[corpus_id]

            
            
            max_chunks = max([len(x) for x in corpus.values()])
            
            corpus = self._flatten_chunks(corpus)   # 'title', 'prev', 'next', 'text'
            # print(corpus)
            
            k_values = self._calculate_k_values(max_chunks)
            # determine the maximum number of documents to consider in a ranking
            max_k = int(max(k_values) / max_chunks)
            retriever = RetrievalEvaluator(
                model,
                k_values=k_values,
                encode_kwargs=(encode_kwargs or dict()),
                **kwargs,
            )
            results = retriever(corpus, queries)    

        elif self.pooling_alg == 'late-chunking':
            query_ids = list(queries.keys())
            query_texts = [queries[k] for k in query_ids]
            if hasattr(model, 'encode_queries'):
                query_embs = model.encode_queries(query_texts)
            else:
                query_embs = model.encode(query_texts)

            corpus_ids = list(corpus.keys())
            corpus_texts = [
                (
                    f"{corpus[k]['title']} {corpus[k]['text']}"
                    if 'title' in corpus[k]
                    else corpus[k]['text']
                )
                for k in corpus_ids
            ]

            chunk_annotations = self._calculate_annotations(model, corpus_texts)

            corpus_embs = []
            with torch.no_grad():
                for inputs in tqdm(
                    self._batch_inputs(
                        list(zip(corpus_texts, chunk_annotations)),
                        batch_size=batch_size,
                    ),
                    total=(len(corpus_texts) // batch_size),
                #    disable=True
                ):
                    if self.model_has_instructions:
                        instr = model.get_instructions()[1]
                    else:
                        instr = ''
                    text_inputs = [instr + x[0] for x in inputs]
                    annotations = [x[1] for x in inputs]
                    model_inputs = self.tokenizer(
                        text_inputs,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=8192,
                    )
                    if model.device.type == 'cuda':
                        model_inputs = {
                            k: v.to(model.device) for k, v in model_inputs.items()
                        }
                    if 'jina' in self.model_name:
                        model_outputs = model(**model_inputs)
                    else:
                        model_outputs = model.model(**model_inputs)
                    output_embs = chunked_pooling(
                        model_outputs, annotations, max_length=8192
                    )
                    corpus_embs.extend(output_embs)
                    
                    
        elif self.pooling_alg == 'long-late-chunking':
            query_ids = list(queries.keys())
            query_texts = [queries[k] for k in query_ids]
            if hasattr(model, 'encode_queries'):
                query_embs = model.encode_queries(query_texts)
            else:
                query_embs = model.encode(query_texts)

            corpus_ids = list(corpus.keys())
            corpus_texts = [
                (
                    f"{corpus[k]['title']} {corpus[k]['text']}"
                    if 'title' in corpus[k]
                    else corpus[k]['text']
                )
                for k in corpus_ids
            ]

            chunk_annotations = self._calculate_annotations(model, corpus_texts)

            corpus_embs = []
            with torch.no_grad():
                for inputs in tqdm(
                    self._batch_inputs(
                        list(zip(corpus_texts, chunk_annotations)),
                        batch_size=batch_size,
                    ),
                    total=(len(corpus_texts) // batch_size),
                #    disable=True
                ):
                    if self.model_has_instructions:
                        instr = model.get_instructions()[1]
                    else:
                        instr = ''
                    text_inputs = [instr + x[0] for x in inputs]
                    annotations = [x[1] for x in inputs]
                    model_inputs = self.tokenizer(
                        text_inputs,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=8192,
                    )
                    if model.device.type == 'cuda':
                        model_inputs = {
                            k: v.to(model.device) for k, v in model_inputs.items()
                        }
                        
                        
                    if self.long_late_chunking_embed_size > 0:
                        model_outputs = self._embed_with_overlap(model, model_inputs)
                        output_embs = chunked_pooling(
                            [model_outputs], annotations, max_length=None
                        )
                    else:  # truncation
                        if 'jina' in self.model_name:
                            model_outputs = model(**model_inputs)
                        else:
                            model_outputs = model.model(**model_inputs)
                        output_embs = chunked_pooling(
                            model_outputs, annotations, max_length=8192
                        )
                    corpus_embs.extend(output_embs)


            max_chunks = max([len(x) for x in corpus_embs])
            k_values = self._calculate_k_values(max_chunks)
            # determine the maximum number of documents to consider in a ranking
            max_k = int(max(k_values) / max_chunks)
            (
                chunk_id_list,
                doc_to_chunk,
                flattened_corpus_embs,
            ) = self.flatten_corpus_embs(corpus_embs, corpus_ids)
            similarity_matrix = np.dot(query_embs, flattened_corpus_embs.T)
            results = self.get_results(
                chunk_id_list, k_values, query_ids, similarity_matrix
            )

        doc_results = self.get_doc_results(results)

        ndcg, _map, recall, precision, _ = RetrievalEvaluator.evaluate(
            relevant_docs,
            doc_results,
            [k for k in k_values if k <= max_k],
            ignore_identical_ids=kwargs.get('ignore_identical_ids', True),
        )
        mrr, _ = RetrievalEvaluator.evaluate_custom(
            relevant_docs,
            doc_results,
            [k for k in k_values if k <= max_k],
            'mrr',
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        self._add_main_score(scores)
        print(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def get_results(self, chunk_id_list, k_values, query_ids, similarity_matrix):
        results = {}
        for i, query_id in tqdm(enumerate(query_ids)):
            query_results = {}
            for idx, score in enumerate(similarity_matrix[i]):
                chunk_id = chunk_id_list[idx]
                query_results[chunk_id] = score
            # Sort results by score and only keep the top k scores
            sorted_query_results = dict(
                sorted(query_results.items(), key=lambda item: item[1], reverse=True)[
                    : max(k_values)
                ]
            )
            results[query_id] = sorted_query_results
        return results

    def flatten_corpus_embs(self, corpus_embs, corpus_ids):
        doc_to_chunk = {}
        flattened_corpus_embs = []
        chunk_id_list = []
        for doc_id, emb in zip(corpus_ids, corpus_embs):
            for i, chunk in enumerate(emb):
                flattened_corpus_embs.append(chunk)
                doc_to_chunk[f"{doc_id}~{i}"] = doc_id
                chunk_id_list.append(f"{doc_id}~{i}")
        flattened_corpus_embs = np.vstack(flattened_corpus_embs)
        flattened_corpus_embs = self._normalize(flattened_corpus_embs)
        return chunk_id_list, doc_to_chunk, flattened_corpus_embs

    @staticmethod
    def get_doc_results(results):
        doc_results = dict()
        for q, result_chunks in results.items():
            docs = dict()
            for c_id, score in result_chunks.items():
                d_id = '~'.join(c_id.split('~')[:-1])
                if (d_id not in docs) or (score > docs[d_id]):
                    docs[d_id] = float(score)
            doc_results[q] = docs
        return doc_results

    def _calculate_k_values(self, max_chunks):
        k_values = [1, 3, 5, 10, 20]
        n = 2
        while 10**n < 100 * max_chunks:
            k_values.append(10**n)
            n += 1
        return k_values

    def _apply_chunking(self, corpus, tokenizer):
        chunked_corpus = dict()
        for k, v in tqdm(corpus.items()):
            text = f"{v['title']} {v['text']}" if 'title' in v else v['text']
            current_doc = []
            chunk_annotations = self.chunker.chunk(
                text,
                tokenizer,
                chunking_strategy=self.chunking_strategy,
                **self.chunking_args,
            )
            tokens = tokenizer.encode_plus(text, add_special_tokens=False)
            for start_token_idx, end_token_idx in chunk_annotations:
                text_chunk = tokenizer.decode(
                    tokens.encodings[0].ids[start_token_idx:end_token_idx]
                )
                current_doc.append({'text': text_chunk})
            chunked_corpus[k] = current_doc
        return chunked_corpus

    def _calculate_annotations(self, model, corpus_texts):
        if self.model_has_instructions:
            instr = model.get_instructions()[1]
            instr_tokens = self.tokenizer(instr, add_special_tokens=False)
            n_instruction_tokens = len(instr_tokens[0])
        else:
            n_instruction_tokens = 0
        chunk_annotations = [
            self._extend_special_tokens(
                self.chunker.chunk(
                    text,
                    self.tokenizer,
                    chunking_strategy=self.chunking_strategy,
                    **self.chunking_args,
                ),
                n_instruction_tokens=n_instruction_tokens,
            )
            for text in corpus_texts
        ]
        return chunk_annotations

    @staticmethod
    def _flatten_chunks(chunked_corpus):
        flattened_corpus = dict()
        for k, li in chunked_corpus.items():
            for i, c in enumerate(li):
                flattened_corpus[f'{k}~{i}'] = c

        return flattened_corpus

    @staticmethod
    def _normalize(x):
        return x / np.linalg.norm(x, axis=1)[:, None]

    @staticmethod
    def _batch_inputs(li, batch_size):
        for i in range(0, len(li), batch_size):
            yield li[i : i + batch_size]

    @staticmethod
    def _extend_special_tokens(
        annotations, n_instruction_tokens=0, include_prefix=True, include_sep=True
    ):
        """Extends the spans because of additional special tokens, e.g. the CLS token
        which are not considered by the chunker.
        """
        new_annotations = []
        for i in range(len(annotations)):
            add_left_offset = 1 if (not include_prefix) or int(i > 0) else 0
            left_offset = 1 + n_instruction_tokens
            left = (
                annotations[i][0] + add_left_offset * left_offset
            )  # move everything by one for [CLS]

            add_sep = 1 if include_sep and ((i + 1) == len(annotations)) else 0
            right_offset = left_offset + add_sep
            right = (
                annotations[i][1] + right_offset
            )  # move everything by one for [CLS] and the last one for [SEP]

            new_annotations.append((left, right))
        return new_annotations

    @staticmethod
    def _prune(queries, corpus, relevant_docs, prune_size):
        new_queries = {'test': {}}
        new_corpus = {'test': {}}
        new_relevant_docs = {'test': {}}
        for i, key in enumerate(relevant_docs['test']):
            if i >= prune_size:
                break
            new_relevant_docs['test'][key] = relevant_docs['test'][key]
            for x in relevant_docs['test'][key]:
                new_corpus['test'][x] = corpus['test'][x]
            new_queries['test'][key] = queries['test'][key]
        return new_queries, new_corpus, new_relevant_docs

    def _calculate_metrics_from_split(*args, **kwargs):
        pass

    def _evaluate_subset(*args, **kwargs):
        pass
