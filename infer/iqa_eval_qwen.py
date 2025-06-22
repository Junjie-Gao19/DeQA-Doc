import argparse
import json
import os
from collections import defaultdict
from io import BytesIO

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # Model
    disable_torch_init()

    # Load Qwen2.5-VL model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.load_bf16 else (torch.float16 if args.load_fp16 else "auto"),
        device_map=args.device if args.device != "auto" else "auto",
        load_in_8bit=args.load_8bit,
        load_in_4bit=args.load_4bit,
        attn_implementation="flash_attention_2" if args.flash_attention else None,
    )
    
    # Load processor and tokenizer
    if args.min_pixels and args.max_pixels:
        processor = AutoProcessor.from_pretrained(
            args.model_path, 
            min_pixels=args.min_pixels, 
            max_pixels=args.max_pixels
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 检查 GPU 使用情况
    if torch.cuda.is_available():
        print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前 GPU 内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        
        # 设置默认设备
        if args.device == "auto":
            args.device = "cuda"
    else:
        print("警告: GPU 不可用，将使用 CPU 运行（会很慢）")
        if args.device == "auto":
            args.device = "cpu"
    
    print(f"使用设备: {args.device}")
    print(f"批处理大小: {args.batch_size}")

    meta_paths = args.meta_paths
    root_dir = args.root_dir
    batch_size = args.batch_size
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with_prob = args.with_prob

    # Qwen2.5-VL conversation template
    inp = "How would you rate the quality of this image?"
    
    # Get token IDs for quality levels
    toks = args.level_names
    print("Quality levels:", toks)
    
    # Tokenize each quality level name separately to get their token IDs
    ids_ = []
    for level in toks:
        full_text = f"The quality of the image is {level}"
        tokens = tokenizer(full_text, add_special_tokens=False).input_ids
        level_text = "The quality of the image is"
        prefix_tokens = tokenizer(level_text, add_special_tokens=False).input_ids
        if len(tokens) > len(prefix_tokens):
            level_id = tokens[len(prefix_tokens)]
            ids_.append(level_id)
            print(f"Level '{level}' token ID: {level_id}")
        else:
            print(f"Level '{level}' not found in the tokenizer vocabulary.")
            ids_.append(tokenizer.unk_token_id)  # Use unknown token ID if not found
    
    print("Token IDs:", ids_)

    for meta_path in meta_paths:
        with open(meta_path) as f:
            iqadata = json.load(f)

        batch_messages = []
        batch_data = []

        imgs_handled = []
        save_path = os.path.join(save_dir, os.path.basename(meta_path))
        if os.path.exists(save_path):
            with open(save_path) as fr:
                for line in fr:
                    meta_res = json.loads(line)
                    imgs_handled.append(meta_res["image"])

        meta_name = os.path.basename(meta_path)
        for i, llddata in enumerate(tqdm(iqadata, desc=f"Evaluating [{meta_name}]")):
            try:
                filename = llddata["image"]
            except:
                filename = llddata["img_path"]
            if filename in imgs_handled:
                continue

            llddata["logits"] = defaultdict(float)
            llddata["probs"] = defaultdict(float)

            image_path = os.path.join(root_dir, filename)
            
            # Prepare message for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": inp},
                    ],
                }
            ]

            batch_messages.append(messages)
            batch_data.append(llddata)

            if (i + 1) % batch_size == 0 or i == len(iqadata) - 1:
                # Process batch - 优化批处理逻辑
                batch_texts = []
                batch_images = []
                
                for messages in batch_messages:
                    # Prepare the text
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    batch_texts.append(text)
                    
                    # Process vision info
                    image_inputs, video_inputs = process_vision_info(messages)
                    if image_inputs:
                        batch_images.extend(image_inputs)
                
                # 真正的批处理
                if batch_texts and batch_images:
                    inputs = processor(
                        text=batch_texts,
                        images=batch_images,
                        padding=True,
                        return_tensors="pt",
                    )
                    
                    # Move to device
                    inputs = inputs.to(model.device)

                    # Run inference
                    with torch.inference_mode():
                        # Forward pass to get logits
                        outputs = model(**inputs)
                        # Get the logits for the last token
                        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                        
                        if with_prob:
                            probs = torch.softmax(logits, dim=-1)

                    # Extract quality level logits/probs
                    for j, xllddata in enumerate(batch_data):
                        for tok, id_ in zip(toks, ids_):
                            xllddata["logits"][tok] = logits[j, id_].item()
                            if with_prob:
                                xllddata["probs"][tok] = probs[j, id_].item()
                        
                        meta_res = {
                            "image": xllddata["image"] if "image" in xllddata else xllddata["img_path"],
                            "logits": dict(xllddata["logits"]),
                        }
                        if with_prob:
                            meta_res["probs"] = dict(xllddata["probs"])
                        
                        with open(save_path, "a") as fw:
                            fw.write(json.dumps(meta_res) + "\n")

                batch_messages = []
                batch_data = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the Qwen2.5-VL model")
    parser.add_argument("--meta-paths", type=str, required=True, nargs="+",
                       help="Paths to metadata JSON files")
    parser.add_argument("--root-dir", type=str, required=True,
                       help="Root directory containing images")
    parser.add_argument("--save-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--level-names", type=str, required=True, nargs="+",
                       help="Quality level names (e.g., 'poor' 'fair' 'good' 'excellent')")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use ('auto', 'cuda', 'cpu', etc.)")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for inference")
    parser.add_argument("--load-8bit", action="store_true",
                       help="Load model in 8-bit precision")
    parser.add_argument("--load-4bit", action="store_true",
                       help="Load model in 4-bit precision")
    parser.add_argument("--load-bf16", action="store_true",
                       help="Load model in BF16 precision")
    parser.add_argument("--load-fp16", action="store_true",
                       help="Load model in FP16 precision")
    parser.add_argument("--flash-attention", action="store_true",
                       help="Enable flash attention 2 for better performance")
    parser.add_argument("--min-pixels", type=int, default=None,
                       help="Minimum pixels for image processing (e.g., 256*28*28)")
    parser.add_argument("--max-pixels", type=int, default=None,
                       help="Maximum pixels for image processing (e.g., 1280*28*28)")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for generation (not used in logits extraction)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum new tokens for generation (not used in logits extraction)")
    parser.add_argument("--with-prob",type=bool, default=False,
                       help="Whether to include probability scores in the output")
    args = parser.parse_args()
    main(args)