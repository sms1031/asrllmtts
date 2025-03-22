import os
import torch
import soundfile as sf
import logging
from datetime import datetime

# 假设 SparkTTS 和相关工具类已经实现
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI


def initialize_model(model_dir="", device=0):
    """加载模型"""
    logging.info(f"加载模型: {model_dir}")
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = SparkTTS(model_dir, device)
    return model


def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
):
    """执行 TTS 推理并保存生成的音频"""
    logging.info(f"保存音频到: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 使用时间戳生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("开始推理...")

    # 执行推理并保存输出音频
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )

        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"音频已保存到: {save_path}")

    return save_path


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 模型路径和设备
    model_dir = "/root/work/work_asrllmtts/model/SparkAudio/Spark-TTS-0___5B"  # 替换为你的模型路径
    device = 0  # 替换为你的 GPU 设备 ID 或设置为 -1 使用 CPU

    # 初始化模型
    model = initialize_model(model_dir=model_dir, device=device)

    # 测试推理
    text = "这是一个测试文本，用于验证模型是否正常工作。"
    prompt_text = None  # 如果不需要提示文本，可以设置为 None
    prompt_speech = None  # 如果不需要提示音频，可以设置为 None
    gender = "male"  # 或 "female"
    pitch = LEVELS_MAP_UI[3]  # 音高，根据 LEVELS_MAP_UI 映射
    speed = LEVELS_MAP_UI[3]  # 语速，根据 LEVELS_MAP_UI 映射

    # 执行推理
    save_path = run_tts(
        text=text,
        model=model,
        prompt_text=prompt_text,
        prompt_speech=prompt_speech,
        gender=gender,
        pitch=pitch,
        speed=speed,
        save_dir="example/results",
    )

    logging.info(f"测试完成，音频已保存到: {save_path}")