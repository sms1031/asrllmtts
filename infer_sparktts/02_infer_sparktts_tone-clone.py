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
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("开始推理...")
    with torch.no_grad():
        # 调整参数以匹配模型的实现
        wav = model.inference(
            text=text,
            prompt_speech=prompt_speech,  # 提示音频
            prompt_text=prompt_text,     # 提示文本
            gender=gender,               # 性别
            pitch=pitch,                 # 音高
            speed=speed,                 # 语速
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"音频已保存到: {save_path}")
    return save_path


def run_voice_cloning(
    text,
    model,
    prompt_text,
    prompt_speech,
    save_dir="example/results",
):
    """执行音色克隆推理"""
    logging.info(f"音色克隆 - 保存音频到: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"cloning_{timestamp}.wav")

    logging.info("开始音色克隆推理...")
    with torch.no_grad():
        # 调整参数以匹配模型的实现
        wav = model.inference(
            text=text,
            prompt_speech_path=prompt_speech,  # 提示音频
            prompt_text=prompt_text,     # 提示文本
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"音色克隆音频已保存到: {save_path}")
    return save_path


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 模型路径和设备
    model_dir = "/root/work/work_asrllmtts/model/SparkAudio/Spark-TTS-0___5B"  # 替换为你的模型路径
    device = 0  # 替换为你的 GPU 设备 ID 或设置为 -1 使用 CPU

    # 初始化模型
    model = initialize_model(model_dir=model_dir, device=device)

    # 音色克隆测试
    logging.info("开始音色克隆测试...")
    text = "这是一个音色克隆测试文本。"
    prompt_text = "这是参考文本，用于音色克隆。"
    prompt_speech = "/root/work/work_asrllmtts/work_spark-tts/Spark-TTS/example/results/tts_20250309150025.wav"  # 替换为你的参考音频路径
    cloning_save_path = run_voice_cloning(
        text=text,
        model=model,
        prompt_text=prompt_text,
        prompt_speech=prompt_speech,
        save_dir="example/results",
    )
    logging.info(f"音色克隆测试完成，音频已保存到: {cloning_save_path}")