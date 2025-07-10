import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainer
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from typing import Optional, Union, Any, Dict



class DeQAScoreLoss(nn.Module):
    def __init__(
        self,
        tokenizer,
        weight_desp=1.0,
        weight_next_token=1.0,
        weight_in_level=1.0,
        weight_softkl=1.0,
        # 定义三个属性的前缀
        level_prefixes={
            "overall_quality": "The overall_quality of the image is",
            "sharpness": "The sharpness of the image is",
            "color_fidelity": "The color_fidelity of the image is"
        },
        level_names=["excellent", "good", "fair", "poor", "bad"],
    ):
        super().__init__()
        self.weight_desp = weight_desp
        self.weight_next_token = weight_next_token
        self.weight_in_level = weight_in_level
        self.weight_softkl = weight_softkl
        
        # 转换level_names为token ids
        self.level_ids = []
        for name in level_names:
            # 使用特殊标记来确保单词被当作一个整体处理
            token_ids = tokenizer(f" {name}", add_special_tokens=False)["input_ids"]
            # 获取最后一个token作为level id
            self.level_ids.append(token_ids[-1])
            print(f"Level name '{name}' tokenized as: {tokenizer.decode([token_ids[-1]])}")
        
        # 转换每个属性的前缀为token ids
        self.level_prefixes = {}
        for attr, prefix in level_prefixes.items():
            self.level_prefixes[attr] = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            print(f"{attr} prefix tokens: {tokenizer.decode(self.level_prefixes[attr])}")
        
        print(f"Level ids: {self.level_ids}")
        print(f"Level names: {[tokenizer.decode([id]) for id in self.level_ids]}")
        
        self.ce_loss = nn.CrossEntropyLoss()

    def find_prefix(self, labels, prefix):
        """找到prefix在labels中的位置"""
        batch_size = labels.shape[0]
        prefix_len = len(prefix)
        indices = []
        
        for i in range(batch_size):
            for j in range(labels.shape[1] - prefix_len + 1):
                if torch.all(labels[i, j:j+prefix_len] == prefix):
                    indices.append(j + prefix_len)
                    break
                    
        return torch.tensor(indices, device=labels.device)

    def softkl_loss(self, logits, labels, level_probs, prefix):
        """计算SoftKL损失"""
        batch_size = logits.shape[0]
        idx_prefix_label = self.find_prefix(labels, prefix)
        idx_level_label = idx_prefix_label
        
        # 获取level token的logits
        logits_level = logits[torch.arange(batch_size), idx_level_label]
        
        # 计算预测的概率分布
        preds = torch.softmax(logits_level, dim=1)
        
        # 创建目标分布
        target = torch.zeros_like(preds)
        target[:, self.level_ids] = level_probs
        target = target.detach()
        
        # 计算KL散度
        pred_log = torch.log(preds)
        loss_kl = F.kl_div(pred_log, target, reduction="batchmean")
        
        return loss_kl, idx_level_label

    def forward(self, model_output, labels, images=None, scores=None):
        """
        Args:
            model_output: 模型输出，包含logits
            labels: 标签
            images: 图像输入
            scores: 包含三个属性评分的字典
                {
                    "overall_quality": {
                        "level_probs": [0.1, 0.2, 0.3, 0.2, 0.2]
                    },
                    "sharpness": {
                        "level_probs": [0.2, 0.3, 0.2, 0.2, 0.1]
                    },
                    "color_fidelity": {
                        "level_probs": [0.1, 0.3, 0.3, 0.2, 0.1]
                    }
                }
        """
        logits = model_output.logits
        
        # 1. 描述任务损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_desp = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 2. 评分任务损失
        loss_score = 0
        loss_details = {}
        
        # 2.1 下一个token预测损失
        loss_next_token = loss_desp
        loss_details["loss_next_token"] = loss_next_token
        
        # 2.2 对每个属性计算SoftKL损失和等级内损失
        if scores is not None:
            for attr, score_info in scores.items():
                if attr in self.level_prefixes:
                    level_probs = torch.tensor(score_info["level_probs"]).to(logits.device)
                    
                    # 计算SoftKL损失
                    loss_kl, idx_level = self.softkl_loss(
                        logits, 
                        labels, 
                        level_probs,
                        self.level_prefixes[attr]
                    )
                    loss_score += self.weight_softkl * loss_kl
                    loss_details[f"{attr}_loss_kl"] = loss_kl
                    
                    # 计算等级内损失
                    probs = torch.softmax(logits[torch.arange(logits.shape[0]), idx_level], dim=1)
                    probs_in_level = probs[:, self.level_ids].sum(dim=1)
                    loss_in_level = torch.max(
                        torch.tensor(1e-2).to(probs_in_level), 
                        1 - probs_in_level.mean()
                    )
                    loss_score += self.weight_in_level * loss_in_level
                    loss_details[f"{attr}_loss_in_level"] = loss_in_level
        
        # 总损失
        total_loss = self.weight_desp * loss_desp + loss_score
        
        loss_details.update({
            "loss": total_loss,
            "loss_desp": loss_desp,
            "loss_score": loss_score
        })
        
        return loss_details
