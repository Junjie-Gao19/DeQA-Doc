import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedTokenizer

def calculate_next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # 将输入序列向右移动一位，使得每个位置预测下一个token
    shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个token的logits
    shift_labels = labels[..., 1:].contiguous()      # 去掉第一个token的label
    
    # 将序列展平，用于计算交叉熵损失
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, logits.size(-1))  # [batch_size * (seq_len-1), vocab_size]
    shift_labels = shift_labels.view(-1)                   # [batch_size * (seq_len-1)]
    
    # 确保标签在正确的设备上
    shift_labels = shift_labels.to(shift_logits.device)
    # 计算交叉熵损失
    loss = loss_fct(shift_logits, shift_labels)
    
    return loss
    
def calculate_single_loss(
    model,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    return_dict: Optional[bool] = None,
    use_softkl_loss: Optional[bool] = None,
    level_probs: Optional[torch.Tensor] = None,
    level_prefix: Optional[str] = "quality of the image is",
    tokenizer: Optional["PreTrainedTokenizer"] = None,
    level_ids: Optional[List[int]] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    计算single形式数据的loss
    
    Args:
        model: 模型实例
        input_ids: 输入token ids
        attention_mask: 注意力掩码
        past_key_values: 过去的key-value缓存
        inputs_embeds: 输入嵌入
        labels: 标签
        use_cache: 是否使用缓存
        output_attentions: 是否输出注意力权重
        output_hidden_states: 是否输出隐藏状态
        images: 图像输入
        image_grid_thw: 图像网格尺寸
        return_dict: 是否返回字典形式
        use_softkl_loss: 是否使用softkl loss
        level_probs: 等级概率
        level_prefix: 等级前缀文本
        tokenizer: 分词器
        level_ids: 等级token ids列表
        
    Returns:
        CausalLMOutputWithPast: 包含loss的输出
    """
    # 获取实际的模型实例（处理DataParallel情况）
    actual_model = model.module if hasattr(model, "module") else model
    
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else actual_model.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else actual_model.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else actual_model.config.use_return_dict
    )
    
    # 获取模型输出
    if images is not None:
        # 使用模型支持的方式处理图像输入
        outputs = actual_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=images,  # 使用pixel_values参数传递图像
            image_grid_thw=image_grid_thw
        )
    else:
        outputs = actual_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    logits = outputs.logits

    loss_kl = None
    if use_softkl_loss and labels is not None:
        loss_kl, idx_level_label, idx_level_logit = softkl_loss(
            logits, 
            labels, 
            level_probs, 
            actual_model,
            level_prefix=level_prefix,
            tokenizer=tokenizer,
            level_ids=level_ids
        )

        def del_elements(source, idx):
            """source: [B, N] / [B, N, V],
            idx: [B, ] with the value range [0, N-1]"""
            mask = torch.ones([*source.shape[:2]], dtype=torch.bool)
            for idx_1, idx_del in enumerate(idx):
                mask[idx_1, idx_del] = False
            if len(source.shape) == 2:
                source_del = source[mask].view(source.size(0), source.size(1)-1)
            else:
                assert len(source.shape) == 3
                source_del = source[mask].view(source.size(0), source.size(1)-1, source.size(2))
            return source_del

        labels_del = del_elements(labels, idx_level_label)
        logits_del = del_elements(logits, idx_level_logit)

    loss = None
    if labels is not None:
        if loss_kl is None:
            loss = calculate_next_token_loss(logits, labels)  # 添加权重系数0.01
        else:
            loss = calculate_next_token_loss(logits_del, labels_del)  # 添加权重系数0.01

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    if loss is not None and loss_kl is not None:
        # 获取weight_softkl，如果不存在则使用默认值1.0
        weight_softkl = getattr(actual_model.config, "weight_softkl", 1.0)
        loss = loss + weight_softkl * loss_kl

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def softkl_loss(logits, labels, level_probs, model, level_prefix: Optional[str] = None, tokenizer: Optional[PreTrainedTokenizer] = None, level_ids: Optional[List[int]] = None):
    """
    计算softkl loss
    
    Args:
        logits: 模型输出的logits
        labels: 标签
        level_probs: 等级概率
        model: 模型实例
        level_prefix: 等级前缀文本
        tokenizer: 分词器
        level_ids: 等级token ids列表
        
    Returns:
        loss_kl: KL散度loss
        idx_level_label: 等级标签的索引
        idx_level_logit: 等级logits的索引
    """
    batch_size = logits.shape[0]
    device = logits.device  # 获取logits所在的设备
    
    #print("前缀是",level_prefix)
    # 如果提供了 level_prefix，则使用它
    if level_prefix is not None and tokenizer is not None:
        # 将 level_prefix 转换为 token ids
        prefix_ids = tokenizer(level_prefix, add_special_tokens=False).input_ids
        level_prefix = torch.tensor(prefix_ids).to(device)
        #print("前缀token为:",level_prefix)
    else:
        # 如果没有提供 level_prefix 或 tokenizer，则使用默认值
        level_prefix = torch.tensor([0]).to(device)  # 使用一个默认的 token id
    #print("label为:",labels)
    idx_prefix_label = find_prefix(labels, level_prefix)  # [B]
    idx_level_label = idx_prefix_label + level_prefix.shape[0]  # [B]
    #print("level_id为",level_ids)
    level_ids_label = labels[torch.arange(batch_size), idx_level_label]  # [B]
    #print("level_ids_label为",level_ids_label)
    for level_id in level_ids_label:
        assert level_id in level_ids

    # After padding in prepare_inputs_labels_for_multimodal(), the length of labels will be the same as logits
    assert logits.shape[1] == labels.shape[1]
    idx_level_logit = idx_level_label - 1  # [B]
    logits_level_ids = logits[
        torch.arange(batch_size), idx_level_logit
    ].contiguous()  # [B, V]

    preds = torch.softmax(logits_level_ids, dim=1)  # [B, V]
    target = torch.zeros_like(preds)  # [B, V]
    
    # 确保level_probs的数据类型和设备与target一致
    if level_probs is not None:
        level_probs = level_probs.to(dtype=target.dtype, device=device)
    
    # 确保level_ids在正确的设备上
    level_ids = torch.tensor(level_ids, device=device)
    
    target[:, level_ids] = level_probs
    target = target.detach()

    pred_log = torch.log(preds)
    loss_kl = F.kl_div(pred_log, target, reduction="batchmean")
    return loss_kl, idx_level_label, idx_level_logit

def find_prefix(labels, prefix):
    """
    在labels中查找prefix的位置
    
    Args:
        labels: 标签序列 [batch_size, seq_len]
        prefix: 要查找的前缀
        
    Returns:
        前缀在labels中的起始位置 [batch_size]
    """
    batch_size = labels.shape[0]
    prefix_len = len(prefix)
    
    # 打印调试信息
    #print("labels shape:", labels.shape)
    #print("prefix:", prefix)
    #print("prefix_len:", prefix_len)
    
    # 为每个样本找到前缀位置
    prefix_positions = []
    for b in range(batch_size):
        found = False
        for i in range(labels.shape[1] - prefix_len + 1):
            # 检查当前位置是否匹配前缀
            matches = True
            for j in range(prefix_len):
                # 跳过IGNORE_INDEX
                if labels[b, i+j] == -100:
                    matches = False
                    break
                if labels[b, i+j] != prefix[j]:
                    matches = False
                    break
            if matches:
                #print(f"样本 {b} 找到前缀，位置: {i}")
                prefix_positions.append(i)
                found = True
                break
        if not found:
            print(f"样本 {b} 未找到前缀")
            prefix_positions.append(0)  # 如果找不到前缀，使用默认位置0
            
    return torch.tensor(prefix_positions, device=labels.device) 