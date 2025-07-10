import argparse
import json
import numpy as np
import os
import random
from scipy.stats import pearsonr, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="Label parameters for DeQA-Score with multiple dimensions (without std)")
    parser.add_argument("--config", type=str, default="./config.json")
    args = parser.parse_args()
    return args


questions = [
    "What do you think about the quality of this image?",
    "Can you rate the quality of this picture?",
    "Can you judge the quality of this image?",
    "How would you rate the quality of this image?",
    "How would you judge the quality of this image?",
    "What is your quality rating for this image?",
    "What's your opinion on the quality of this picture?",
    "Rate the quality of this image.",
    "Could you evaluate the quality of this image?",
    "How do you assess the quality of this image?",
]


def calculate_srcc_plcc(pred, mos):
    """
    计算Spearman秩相关系数（SRCC）和Pearson线性相关系数（PLCC）。
    在计算PLCC之前，进行三阶多项式回归拟合。

    :param pred: 预测分数列表
    :param mos: 真实分数列表
    :return: srcc, plcc
    """
    srcc, _ = spearmanr(pred, mos)
    try:
        coeffs = np.polyfit(pred, mos, 3)
        poly = np.poly1d(coeffs)
        pred_mapped = poly(pred)
        plcc, _ = pearsonr(pred_mapped, mos)
    except Exception as e:
        print(f"Error in polynomial regression: {e}")
        plcc = np.nan
    return srcc, plcc


def get_level(mos, min_mos, max_mos):
    """
    根据MOS值获取对应的质量等级文本。

    :param mos: 归一化后的MOS值
    :param min_mos: 评分维度的最小值
    :param max_mos: 评分维度的最大值
    :return: 对应的质量等级文本
    """
    eps = 1e-8
    texts = ["bad", "poor", "fair", "good", "excellent"]
    for idx in range(1, len(texts) + 1):
        mos_left = min_mos + (idx - 1) / 5 * (max_mos - min_mos) - eps
        mos_right = min_mos + idx / 5 * (max_mos - min_mos) + eps
        if mos > mos_left and mos <= mos_right:
            level = idx
            break
    text = texts[level - 1]
    return text


def get_binary_probs(mos, min_mos=1.0, max_mos=5.0):
    """
    根据MOS值生成二进制概率分布。

    :param mos: 归一化后的MOS值
    :param min_mos: 评分维度的最小值
    :param max_mos: 评分维度的最大值
    :return: 二进制概率分布列表
    """
    eps = 1e-8
    probs = [0, 0, 0, 0, 0]
    for idx in range(1, len(probs)):
        mos_left = min_mos + (idx - 1) / 4 * (max_mos - min_mos) - eps
        mos_right = min_mos + idx / 4 * (max_mos - min_mos) + eps
        if mos > mos_left and mos <= mos_right:
            probs[idx - 1] = (mos_right - mos) / (mos_right - mos_left)
            probs[idx] = (mos - mos_left) / (mos_right - mos_left)
            break
    probs = probs[::-1]  # 从 "excellent" 到 "bad"
    return probs

def convert_to_native(obj):
    """Recursively convert numpy data types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj

def main(cfg):
    density_type = cfg.get("density_type", "pdf")  # 默认使用 "pdf"
    # with open(cfg["split_json"]) as fr:
    #     split = json.load(fr)
    with open(cfg["mos_json"]) as fr:
        mos_dict = json.load(fr)
    save_train = cfg["save_train"]
    save_test = cfg["save_test"]

    # 定义评分维度
    scores_list = ['overall_quality', 'sharpness', 'color_fidelity']
    moses = {score: [] for score in scores_list}
    imgs = []
    for img in mos_dict:
        imgs.append(img)
        for score in scores_list:
            moses[score].append(float(mos_dict[img][score]))
    # 计算每个评分维度的最小值和最大值
    max_mos = {score: max(moses[score]) for score in scores_list}
    min_mos = {score: min(moses[score]) for score in scores_list}

    num_binary = 0
    # 为每个维度分别存储预测值和实际值
    preds = {score: [] for score in scores_list}
    gts = {score: [] for score in scores_list}
    diffs = {score: [] for score in scores_list}
    train_metas, test_metas = [], []

    # 创建图像到索引的映射，避免在循环中频繁调用 imgs.index(img)
    img_to_index = {img: idx for idx, img in enumerate(imgs)}

    for img in imgs:
        basename = os.path.basename(img)
        training = True

        index = img_to_index[img]

        meta = {
            "id": basename,
            "image": mos_dict[img]["res"],
            "scores": {},
        }

        conversations = []
        # 对每个评分维度进行处理
        for score in scores_list:
            mos = moses[score][index]
            current_max = max_mos[score]
            current_min = min_mos[score]

            # 归一化
            mos_norm = 4 * (mos - 1) / (5 - 1) + 1  # [1, 5]

            # 生成二进制概率分布
            probs_norm = get_binary_probs(mos_norm)
            mos_rec = np.inner(np.array(probs_norm), np.array([5, 4, 3, 2, 1])) 
            diff = abs(mos_rec - mos_norm)

            # 存储预测值和实际值
            preds[score].append(mos_rec)
            gts[score].append(mos_norm)
            diffs[score].append(diff)
            num_binary += 1  # 每个评分维度都使用二进制概率

            # 存储当前评分维度的信息到meta
            meta["scores"][score] = {
                "gt_score": mos,
                "gt_core_norm": mos_norm,
                "level_probs": probs_norm,
            }
            # 构建对话内容（使用二进制概率，无需条件判断）
            if training:
                text = get_level(mos, current_min, current_max)
                query = random.choice(questions)
                resp = cfg["answer"].replace("{}", text)
                conv = {
                    "from": "human",
                    "value": f"{query}\n<|image|>",
                }, {
                    "from": "gpt",
                    "value": resp,
                }
                conversations.extend(conv)

        # 根据是否为训练集或测试集，更新meta
        if training:
            meta["conversations"] = conversations
            train_metas.append(meta)
        else:
            test_metas.append(meta)

    # 保存训练集和测试集
    print("=" * 100)
    print(f"保存 {len(train_metas)} 条记录到 {save_train}")
    native_train_metas = convert_to_native(train_metas)
    with open(save_train, "w", encoding='utf-8') as fw:
        fw.write(json.dumps(native_train_metas, indent=4, ensure_ascii=False))

    print(f"保存 {len(test_metas)} 条记录到 {save_test}")
    native_test_metas = convert_to_native(test_metas)
    with open(save_test, "w", encoding='utf-8') as fw:
        fw.write(json.dumps(native_test_metas, indent=4, ensure_ascii=False))

    # 计算并打印统计指标
    Scoreoverall_quality = 0.0
    Scoresharpness = 0.0
    Scorecolor_fidelity = 0.0

    for score in scores_list:
        srcc, plcc = calculate_srcc_plcc(preds[score], gts[score])
        combined_score = 0.5 * srcc + 0.5 * plcc
        if score == 'overall_quality':
            Scoreoverall_quality = combined_score
        elif score == 'sharpness':
            Scoresharpness = combined_score
        elif score == 'color_fidelity':
            Scorecolor_fidelity = combined_score

        print(f"评分维度: {score}")
        print("  SROCC:", srcc, "PLCC:", plcc)
        print("  综合评分 (0.5 * SROCC + 0.5 * PLCC):", combined_score)
        print("  [差异]")
        print("    L1:", sum(diffs[score]) / len(diffs[score]))
        print("    L2:", np.sqrt((np.array(diffs[score])**2).mean()))
        print()

    print("总体评分:")
    print(f"  Scoreoverall_quality = {Scoreoverall_quality}")
    print(f"  Scoresharpness = {Scoresharpness}")
    print(f"  Scorecolor_fidelity = {Scorecolor_fidelity}")
    print("binary / all:", num_binary, "/", len(train_metas) + len(test_metas))


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding='utf-8') as fr:
        cfg = json.load(fr)
    dataset='diqa'
    main(cfg["dataset_params"][dataset])
