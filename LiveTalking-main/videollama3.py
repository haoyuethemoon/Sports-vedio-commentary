import torch
import time
from transformers import AutoModelForCausalLM, AutoProcessor
import os
from logger import logger
from basereal import BaseReal

def vllm(file,message):
    model_path = "D:\GraduationProject\VideoLLaMA3-main\models\VideoLLaMA3-7B"
    file_path = r'D:\GraduationProject\VideoLLaMA3-main\assets\commentary.txt'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


    @torch.inference_mode()
    def infer(conversation):
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        st = time.time()
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        et = time.time()
        print("本次解析用时{}秒".format(et - st))
        return response

    # Video conversation
    conversation = [
        {"role": "system", "content": "You are a professional commentator."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": "D:/GraduationProject/VideoLLaMA3-main/assets/"+file, "fps": 1, "max_frames": 180}},
                {"type": "text", "text": "视频中的女人在做什么？请用一段专业的中文文字解说动作信息"},
            ]
        },
    ]
    text=infer(conversation)
    print(text)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"文件已保存到：{file_path}")


