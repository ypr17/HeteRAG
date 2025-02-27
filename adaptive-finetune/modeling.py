import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertModel
from jina_bert_implementation.modeling_bert import JinaBertModel
import numpy as np
from scipy import stats

from ark_nlp.nn import Bert
from ark_nlp.factory.task.base._sequence_classification import SequenceClassificationTask



class CoSENT(Bert):
    """
    CoSENT模型

    Args:
        config:
            模型的配置对象
        encoder_trained (:obj:`bool`, optional, defaults to True):
            bert参数是否可训练，默认可训练
        pooling (:obj:`str`, optional, defaults to "last_avg"):
            bert输出的池化方式，默认为"last_avg"，
            可选有["cls", "cls_with_pooler", "first_last_avg", "last_avg", "last_2_avg"]
        dropout (:obj:`float` or :obj:`None`, optional, defaults to None):
            dropout比例，默认为None，实际设置时会设置成0
        output_emb_size (:obj:`int`, optional, defaults to 0):
            输出的矩阵的维度，默认为0，即不进行矩阵维度变换

    Reference:
        [1] https://kexue.fm/archives/8847
        [2] https://github.com/bojone/CoSENT 
    """  # noqa: ignore flake8"

    def __init__(
        self,
        config,
        model_path,
        encoder_trained=True,
        pooling='last_avg',
        dropout=None,
        output_emb_size=0
    ):

        super(CoSENT, self).__init__(config)

        if 'jina' in model_path:
            self.bert = JinaBertModel.from_pretrained(model_path).cuda()
        else:
            self.bert = BertModel.from_pretrained(model_path).cuda()
        self.pooling = pooling

        self.dropout = nn.Dropout(dropout if dropout is not None else 0)

        # if output_emb_size is greater than 0, then add Linear layer to reduce embedding_size,
        # we recommend set output_emb_size = 256 considering the trade-off beteween
        # recall performance and efficiency
        self.output_emb_size = output_emb_size
        if self.output_emb_size > 0:
            self.emb_reduce_linear = nn.Linear(
                config.hidden_size,
                self.output_emb_size
            )
            torch.nn.init.trunc_normal_(
                self.emb_reduce_linear.weight,
                std=0.02
            )

        for param in self.bert.parameters():
            param.requires_grad = encoder_trained

        self.init_weights()

    def get_pooled_embedding(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True
        )

        encoder_feature = self.get_encoder_feature(
            outputs,
            attention_mask
        )

        if self.output_emb_size > 0:
            encoder_feature = self.emb_reduce_linear(encoder_feature)

        encoder_feature = self.dropout(encoder_feature)
        out = F.normalize(encoder_feature, p=2, dim=-1)

        return out

    def cosine_sim(
        self,
        input_ids_a,
        input_ids_b,
        token_type_ids_a=None,
        position_ids_ids_a=None,
        attention_mask_a=None,
        token_type_ids_b=None,
        position_ids_b=None,
        attention_mask_b=None,
        **kwargs
    ):

        query_cls_embedding = self.get_pooled_embedding(
            input_ids_a,
            token_type_ids_a,
            position_ids_ids_a,
            attention_mask_a
        )

        title_cls_embedding = self.get_pooled_embedding(
            input_ids_b,
            token_type_ids_b,
            position_ids_b,
            attention_mask_b
        )

        cosine_sim = torch.sum(
            query_cls_embedding * title_cls_embedding,
            axis=-1
        )

        return cosine_sim

    def forward(
        self,
        input_ids_a,
        input_ids_b,
        token_type_ids_a=None,
        position_ids_ids_a=None,
        attention_mask_a=None,
        token_type_ids_b=None,
        position_ids_b=None,
        attention_mask_b=None,
        label_ids=None,
        **kwargs
    ):

        cls_embedding_a = self.get_pooled_embedding(
            input_ids_a,
            token_type_ids_a,
            position_ids_ids_a,
            attention_mask_a
        )

        cls_embedding_b = self.get_pooled_embedding(
            input_ids_b,
            token_type_ids_b,
            position_ids_b,
            attention_mask_b
        )

        cosine_sim = torch.sum(cls_embedding_a * cls_embedding_b, dim=1) * 20
        cosine_sim = cosine_sim[:, None] - cosine_sim[None, :]
        
        labels = label_ids[:, None] < label_ids[None, :]
        labels = labels.long()
        
        cosine_sim = cosine_sim - (1 - labels) * 1e12
        cosine_sim = torch.cat((torch.zeros(1).to(cosine_sim.device), cosine_sim.view(-1)), dim=0)
        loss = torch.logsumexp(cosine_sim.view(-1), dim=0)

        return cosine_sim, loss
    
class CoSENTTask(SequenceClassificationTask):
    """
    用于CoSENT模型文本匹配任务的Task
    
    Args:
        module: 深度学习模型
        optimizer: 训练模型使用的优化器名或者优化器对象
        loss_function: 训练模型使用的损失函数名或损失函数对象
        class_num (:obj:`int` or :obj:`None`, optional, defaults to None): 标签数目
        scheduler (:obj:`class`, optional, defaults to None): scheduler对象
        n_gpu (:obj:`int`, optional, defaults to 1): GPU数目
        device (:obj:`class`, optional, defaults to None): torch.device对象，当device为None时，会自动检测是否有GPU
        cuda_device (:obj:`int`, optional, defaults to 0): GPU编号，当device为None时，根据cuda_device设置device
        ema_decay (:obj:`int` or :obj:`None`, optional, defaults to None): EMA的加权系数
        **kwargs (optional): 其他可选参数
    """  # noqa: ignore flake8"

    def _on_evaluate_begin_record(self, **kwargs):

        self.evaluate_logs['eval_loss'] = 0
        self.evaluate_logs['eval_step'] = 0
        self.evaluate_logs['eval_example'] = 0

        self.evaluate_logs['labels'] = []
        self.evaluate_logs['eval_sim'] = []

    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():
            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)
            self.evaluate_logs['eval_loss'] += loss.item()

            if 'label_ids' in inputs:
                cosine_sim = self.module.cosine_sim(**inputs).cpu().numpy()
                self.evaluate_logs['eval_sim'].append(cosine_sim)
                self.evaluate_logs['labels'].append(inputs['label_ids'].cpu().numpy())

        self.evaluate_logs['eval_example'] += logits.shape[0]
        self.evaluate_logs['eval_step'] += 1

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        **kwargs
    ):

        if is_evaluate_print:
            if 'labels' in self.evaluate_logs:
                _sims = np.concatenate(self.evaluate_logs['eval_sim'], axis=0)
                _labels = np.concatenate(self.evaluate_logs['labels'], axis=0)
                spearman_corr = stats.spearmanr(_labels, _sims).correlation
                print('evaluate spearman corr is:{:.4f}, evaluate loss is:{:.6f}'.format(
                    spearman_corr,
                    self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step']
                    )
                )
            else:
                print('evaluate loss is:{:.6f}'.format(self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step']))