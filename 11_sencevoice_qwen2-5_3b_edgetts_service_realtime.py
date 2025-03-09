import flask
from flask import Flask, request, jsonify
import wave
import numpy as np
import time
import webrtcvad
import os
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import edge_tts
import asyncio
import langid
import json  # Added for JSON serialization
from datetime import datetime  # Added for timestamps

# --- 配置huggingFace国内镜像 ---
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 参数设置
AUDIO_RATE = 16000        # 音频采样率
AUDIO_CHANNELS = 1        # 单声道
VAD_MODE = 3              # VAD 模式 (0-3, 数字越大越敏感)
FRAME_DURATION = 30       # VAD 帧长度（毫秒），WebRTC VAD要求为10, 20或30ms
FRAMES_PER_BUFFER = int(AUDIO_RATE * FRAME_DURATION / 1000)  # 每个缓冲区中的帧数
OUTPUT_DIR = "./output/output02_01"   # 输出目录
folder_path = "./Test_QWen2_VL/"
audio_file_count = 0

# 会话状态
sessions = {}  # 存储每个会话的状态

# 新增：会话历史记录目录
HISTORY_DIR = "./conversation_history"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)  # 确保历史记录目录存在

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)

# 创建Flask应用
app = Flask(__name__)

# 初始化模型（在全局范围内加载一次）
def load_models():
    global model_senceVoice, model, tokenizer
    
    # -------- SenceVoice 语音识别 --模型加载-----
    from funasr import AutoModel
    model_dir = r"/root/work/work_asrllmtts/model/iic/SenseVoiceSmall"
    model_senceVoice = AutoModel(model=model_dir, trust_remote_code=True)

    # --- QWen2.5大语言模型 ---
    model_name = r"/root/deepseek_r1_train/Qwen/Qwen2___5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return "Models loaded successfully"

# 会话类，用于管理每个客户端的语音数据和VAD状态
class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.dir = os.path.join(OUTPUT_DIR, f"session_{session_id}")
        os.makedirs(self.dir, exist_ok=True)
        
        # 语音活动检测状态
        self.is_speech_active = False
        self.speech_buffer = []  # 存储当前语音片段的帧
        self.silence_counter = 0  # 沉默帧计数器
        self.vad_buffer = b""  # 用于VAD处理的缓冲区
        
        # 当前正在处理的语音片段文件
        self.current_segment_file = None
        self.current_segment_wav = None
        self.segment_counter = 0
        
        # 已完成的语音片段
        self.completed_segments = []
        
        # 当前是否正在处理语音
        self.is_processing = False
        
        # 用于存储处理结果的列表
        self.results_history = []
        
        # 是否正在播放音频（用于打断机制）
        self.is_playing_audio = False
        
        # 新增：完整的对话历史记录
        self.conversation_history = []
        
        # 创建会话历史文件
        self.history_file = os.path.join(HISTORY_DIR, f"session_{session_id}_history.json")
        self.save_conversation_history()  # 初始化空的历史记录文件

    # 新增：保存对话历史记录到文件
    def save_conversation_history(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

    # 新增：添加对话到历史记录
    def add_to_history(self, result):
        # 添加时间戳
        history_item = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_input': result['input_text'],
            'ai_response': result['output_text'],
            'audio_path': result.get('audio_url', ''),
            'language': result.get('language', '')
        }
        
        # 添加到历史记录列表
        self.conversation_history.append(history_item)
        
        # 保存到文件
        self.save_conversation_history()

# 为新会话创建唯一标识符和目录
@app.route('/start_session', methods=['GET'])
def start_session():
    session_id = str(int(time.time()))
    sessions[session_id] = Session(session_id)
    return jsonify({"session_id": session_id})

# 处理音频块并进行VAD检测
@app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    session_id = request.form.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    
    # 获取音频数据
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty audio chunk'}), 400
    
    # 读取WAV文件数据，跳过头部
    audio_data = audio_file.read()
    
    # 如果是WAV文件，需要跳过头部44字节
    if audio_data[:4] == b'RIFF':
        pcm_data = audio_data[44:]
    else:
        pcm_data = audio_data
    
    # 检查是否有语音，如果有且正在播放音频，则触发打断机制
    has_speech = check_for_speech(pcm_data)
    if has_speech and session.is_playing_audio:
        # 打断正在播放的音频
        return jsonify({"status": "interrupt", "message": "User is speaking, stop audio playback"})
    
    # 如果当前正在处理语音，暂时不进行新的处理
    if session.is_processing:
        return jsonify({"status": "processing", "message": "Currently processing previous audio segment"})
    
    # 处理音频数据进行VAD
    result = process_audio_chunk(session, pcm_data)
    
    # 如果有新的处理结果，返回给前端
    if result and 'result' in result:
        return jsonify(result)
    
    return jsonify({"status": "success"})

# 检查音频数据中是否包含语音
def check_for_speech(pcm_data):
    """快速检查音频数据是否包含语音，用于打断机制"""
    frame_bytes = FRAMES_PER_BUFFER * 2  # 16位PCM，每个采样2字节
    
    # 只检查前几帧
    frames_to_check = min(5, len(pcm_data) // frame_bytes)
    speech_detected = False
    
    for i in range(frames_to_check):
        if i * frame_bytes + frame_bytes > len(pcm_data):
            break
            
        frame = pcm_data[i * frame_bytes:(i + 1) * frame_bytes]
        try:
            if vad.is_speech(frame, AUDIO_RATE):
                speech_detected = True
                break
        except:
            continue
    
    return speech_detected

# 设置音频播放状态
@app.route('/audio_status', methods=['POST'])
def audio_status():
    """更新音频播放状态（开始/停止）"""
    session_id = request.form.get('session_id')
    status = request.form.get('status')  # 'playing' 或 'stopped'
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    
    if status == 'playing':
        session.is_playing_audio = True
    elif status == 'stopped':
        session.is_playing_audio = False
    
    return jsonify({"status": "success"})

# 处理音频块进行VAD检测
def process_audio_chunk(session, pcm_data):
    # 将PCM数据添加到VAD缓冲区
    session.vad_buffer += pcm_data
    
    # 每次处理FRAME_DURATION长度的帧
    frame_bytes = FRAMES_PER_BUFFER * 2  # 16位PCM，每个采样2字节
    result = None
    
    # 提取完整的帧进行处理
    while len(session.vad_buffer) >= frame_bytes:
        # 获取一帧数据
        frame = session.vad_buffer[:frame_bytes]
        session.vad_buffer = session.vad_buffer[frame_bytes:]
        
        # 使用VAD检测是否有语音
        try:
            is_speech = vad.is_speech(frame, AUDIO_RATE)
        except:
            # 如果帧无效，跳过
            continue
        
        if is_speech:
            # 检测到语音
            if not session.is_speech_active:
                # 语音开始，创建新的语音片段文件
                start_new_speech_segment(session)
            
            # 重置沉默计数器
            session.silence_counter = 0
            
            # 将帧添加到当前语音片段
            add_frame_to_segment(session, frame)
        else:
            # 没有检测到语音
            if session.is_speech_active:
                # 增加沉默计数器
                session.silence_counter += 1
                
                # 将帧添加到当前语音片段（保留一些沉默以避免截断）
                add_frame_to_segment(session, frame)
                
                # 如果沉默持续时间超过阈值（例如1000ms），结束当前语音片段并自动处理
                if session.silence_counter > (1000 // FRAME_DURATION):  # 1000ms的沉默
                    # 结束当前语音片段
                    end_speech_segment(session)
                    
                    # 如果片段有效，自动处理它
                    if session.completed_segments and not session.is_processing:
                        # 标记正在处理
                        session.is_processing = True
                        
                        # 使用线程异步处理音频，避免阻塞主线程
                        threading.Thread(
                            target=process_completed_segment,
                            args=(session,)
                        ).start()
            # 如果不处于语音活动状态，忽略该帧
    
    # 如果有新的处理结果，返回它
    if session.results_history and session.is_processing == False:
        result = {"status": "new_result", "result": session.results_history[-1]}
        # 清除已返回的结果
        session.results_history = []
    
    return result

# 异步处理完成的语音片段
def process_completed_segment(session):
    try:
        # 合并音频片段
        global audio_file_count
        audio_file_count += 1
        merged_filename = f"audio_{audio_file_count}.wav"
        merged_filepath = os.path.join(OUTPUT_DIR, merged_filename)
        
        # 合并WAV文件
        merge_wav_files(session.completed_segments, merged_filepath)
        
        # 处理合并后的音频文件
        result = process_audio_file(merged_filepath)
        
        # 只有当结果不为None时（通过了文本长度检查）才保存结果
        if result is not None:
            # 保存处理结果以在下一次上传时返回
            session.results_history.append(result)
            
            # 新增：添加到对话历史记录
            session.add_to_history(result)
        
        # 清理会话数据
        session.completed_segments = []
    finally:
        # 标记处理完成
        session.is_processing = False

# 开始新的语音片段
def start_new_speech_segment(session):
    session.is_speech_active = True
    session.silence_counter = 0
    session.speech_buffer = []
    
    # 创建新的WAV文件
    segment_path = os.path.join(session.dir, f"segment_{session.segment_counter}.wav")
    session.current_segment_file = segment_path
    
    # 打开WAV文件以写入
    session.current_segment_wav = wave.open(segment_path, 'wb')
    session.current_segment_wav.setnchannels(AUDIO_CHANNELS)
    session.current_segment_wav.setsampwidth(2)  # 16位PCM
    session.current_segment_wav.setframerate(AUDIO_RATE)

# 添加帧到当前语音片段
def add_frame_to_segment(session, frame):
    if session.current_segment_wav:
        session.current_segment_wav.writeframes(frame)
        session.speech_buffer.append(frame)

# 结束语音片段
def end_speech_segment(session):
    if session.current_segment_wav:
        session.current_segment_wav.close()
        session.current_segment_wav = None
        
        # 只有当语音片段足够长时才保留（至少1秒）
        min_frames = 500 // FRAME_DURATION  # 1000ms = 1秒
        if len(session.speech_buffer) >= min_frames:
            session.completed_segments.append(session.current_segment_file)
        else:
            # 如果片段太短，删除文件
            print(f"语音片段太短（小于0.5秒），丢弃: {session.current_segment_file}")
            os.remove(session.current_segment_file)
        
        session.segment_counter += 1
    
    session.is_speech_active = False
    session.speech_buffer = []

# 合并WAV文件
def merge_wav_files(wav_files, output_path):
    """合并多个WAV文件成一个WAV文件"""
    try:
        # 读取第一个WAV文件以获取格式信息
        with wave.open(wav_files[0], 'rb') as w:
            params = w.getparams()
        
        # 创建输出WAV文件
        with wave.open(output_path, 'wb') as output:
            output.setparams(params)
            
            # 逐个读取并写入每个WAV文件的数据部分
            for wav_file in wav_files:
                with wave.open(wav_file, 'rb') as w:
                    output.writeframes(w.readframes(w.getnframes()))
        
        return True
    except Exception as e:
        print(f"合并WAV文件失败: {e}")
        # 如果合并失败，创建一个空WAV文件
        create_empty_wav(output_path)
        return False

# 创建空白WAV文件（作为备选）
def create_empty_wav(file_path):
    # 创建1秒的静音WAV文件
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(AUDIO_RATE)
        # 1秒的静音
        silence = np.zeros(AUDIO_RATE, dtype=np.int16)
        wf.writeframes(silence.tobytes())

# # 处理音频文件
# def process_audio_file(audio_path):
#     # -------- SenceVoice 推理 ---------

#     res = model_senceVoice.generate(
#         input=audio_path,
#         cache={},
#         language="auto",
#         use_itn=False,
#     )
#     recognized_text = res[0]['text'].split(">")[-1]
    
#     # 检查识别文本长度，如果小于2个字符，则返回None
#     if len(recognized_text.strip()) < 2:
#         print(f"识别文本太短（小于2个字符），丢弃: '{recognized_text}'")
#         return None
        
#     prompt = recognized_text + "，回答简短一些，保持200字以内！"
#     print("ASR OUT:", prompt)
    
#     # -------- 模型推理阶段，将语音识别结果作为大模型Prompt ------
#     messages = [
#         {"role": "system", "content": "你叫千问，是一个18岁的女大学生，性格活泼开朗，说话俏皮"},
#         {"role": "user", "content": prompt},
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512,
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print("answer", output_text)

#     # 语种识别 -- langid
#     language, confidence = langid.classify(output_text)

#     language_speaker = {
#         "ja": "ja-JP-NanamiNeural",
#         "fr": "fr-FR-DeniseNeural",
#         "es": "ca-ES-JoanaNeural",
#         "de": "de-DE-KatjaNeural",
#         "zh": "zh-CN-XiaoyiNeural",
#         "en": "en-US-AnaNeural",
#     }

#     if language not in language_speaker.keys():
#         used_speaker = "zh-CN-XiaoyiNeural"
#     else:
#         used_speaker = language_speaker[language]
#         print("检测到语种：", language, "使用音色：", language_speaker[language])

#     # 生成语音回复
#     global audio_file_count
#     tts_filename = f"sft_{audio_file_count}.mp3"
#     tts_filepath = os.path.join(folder_path, tts_filename)
    
#     asyncio.run(generate_speech(output_text, used_speaker, tts_filepath))
    
#     # 返回处理结果
#     return {
#         'input_text': recognized_text,
#         'output_text': output_text,
#         'audio_url': f'/audio/{tts_filename}',
#         'language': language
#     }

def process_audio_file(audio_path):
    total_start = time.perf_counter()  # 总耗时统计起点
    time_stats = {}  # 存储各阶段耗时
    
    # -------- SenceVoice 推理 ---------
    asr_start = time.perf_counter()
    res = model_senceVoice.generate(
        input=audio_path,
        cache={},
        language="auto",
        use_itn=False,
    )
    time_stats['asr'] = time.perf_counter() - asr_start  # ASR耗时
    
    recognized_text = res[0]['text'].split(">")[-1]
    if len(recognized_text.strip()) < 2:
        print(f"识别文本太短（小于2个字符），丢弃: '{recognized_text}'")
        return None
        
    prompt = recognized_text + "，回答简短一些，保持200字以内！"
    print("ASR OUT:", prompt)
    
    # -------- 大模型推理 ---------
    llm_start = time.perf_counter()
    messages = [
        {"role": "system", "content": "你叫千问，是一个18岁的女大学生，性格活泼开朗，说话俏皮"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    time_stats['llm'] = time.perf_counter() - llm_start  # LLM耗时
    
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("answer", output_text)

    # -------- 语种识别 ---------
    lang_start = time.perf_counter()
    language, confidence = langid.classify(output_text)
    time_stats['lang'] = time.perf_counter() - lang_start  # 语种识别耗时

    language_speaker = {
        "ja": "ja-JP-NanamiNeural",
        "fr": "fr-FR-DeniseNeural",
        "es": "ca-ES-JoanaNeural",
        "de": "de-DE-KatjaNeural",
        "zh": "zh-CN-XiaoyiNeural",
        "en": "en-US-AnaNeural",
    }

    if language not in language_speaker.keys():
        used_speaker = "zh-CN-XiaoyiNeural"
    else:
        used_speaker = language_speaker[language]
        print("检测到语种：", language, "使用音色：", language_speaker[language])

    # -------- 语音生成 ---------
    tts_start = time.perf_counter()
    global audio_file_count
    tts_filename = f"sft_{audio_file_count}.mp3"
    tts_filepath = os.path.join(folder_path, tts_filename)
    
    asyncio.run(generate_speech(output_text, used_speaker, tts_filepath))
    time_stats['tts'] = time.perf_counter() - tts_start  # TTS耗时
    
    # -------- 总耗时计算 ---------
    time_stats['total'] = time.perf_counter() - total_start
    
    # 打印耗时统计（保留3位小数）
    print("\n[耗时统计]")
    print(f"ASR识别: {time_stats['asr']:.3f}s")
    print(f"大模型生成: {time_stats['llm']:.3f}s") 
    print(f"语种识别: {time_stats['lang']:.3f}s")
    print(f"语音合成: {time_stats['tts']:.3f}s")
    print(f"总耗时: {time_stats['total']:.3f}s\n")

        # 返回处理结果
    return {
        'input_text': recognized_text,
        'output_text': output_text,
        'audio_url': f'/audio/{tts_filename}',
        'language': language
    }

    


# 生成语音
async def generate_speech(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

# 提供语音文件访问
@app.route('/audio/<filename>')
def serve_audio(filename):
    return flask.send_from_directory(folder_path, filename)

# 新增：获取会话历史记录
@app.route('/get_history', methods=['GET'])
def get_history():
    session_id = request.args.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    
    # 返回会话历史记录
    return jsonify({
        "session_id": session_id,
        "history": session.conversation_history
    })

# 手动结束会话（可选）
@app.route('/end_session', methods=['POST'])
def end_session():
    session_id = request.form.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    # 确保所有对话历史已保存
    sessions[session_id].save_conversation_history()
    
    # 可选：清理会话资源
    # del sessions[session_id]
    
    return jsonify({"status": "session ended"})

# 主页 - 更新后的前端版本，支持持续录音和结果显示，以及打断功能
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>语音助手服务</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; margin-right: 10px; }
            button:hover { background: #45a049; }
            #status, #result { margin-top: 20px; }
            .recording { color: red; }
            #resultContainer { margin-top: 20px; max-height: 400px; overflow-y: auto; }
            .result-item { border-bottom: 1px solid #ccc; padding: 10px 0; }
            .history-btn { background: #2196F3; }
            .history-btn:hover { background: #0b7dda; }
            #historyContainer { margin-top: 20px; display: none; }
            .history-item { padding: 10px; margin-bottom: 5px; background-color: #f5f5f5; border-radius: 5px; }
            .user-text { color: #2196F3; }
            .ai-text { color: #4CAF50; }
        </style>
    </head>
    <body>
        <h1>语音助手服务</h1>
        <p>点击下方按钮开始持续录音，与AI助手对话。说话后停顿会自动处理。说话会自动打断AI正在播放的声音。</p>
        
        <div>
            <button id="recordButton">开始录音</button>
            <button id="historyButton" class="history-btn">查看历史记录</button>
        </div>
        <div id="status"></div>
        <div id="resultContainer"></div>
        <div id="historyContainer"></div>
        
        <script>
            const recordButton = document.getElementById('recordButton');
            const historyButton = document.getElementById('historyButton');
            const statusDiv = document.getElementById('status');
            const resultContainer = document.getElementById('resultContainer');
            const historyContainer = document.getElementById('historyContainer');
            
            let audioContext;
            let recorder;
            let isRecording = false;
            let sessionId = null;
            let uploadInterval;
            let activeAudioElement = null;
            
            recordButton.addEventListener('click', toggleRecording);
            historyButton.addEventListener('click', toggleHistory);
            
            async function toggleRecording() {
                if (isRecording) {
                    await stopRecording();
                } else {
                    await startRecording();
                }
            }
            
            async function toggleHistory() {
                if (historyContainer.style.display === 'block') {
                    historyContainer.style.display = 'none';
                    historyButton.textContent = '查看历史记录';
                } else {
                    if (sessionId) {
                        await loadHistory();
                        historyContainer.style.display = 'block';
                        historyButton.textContent = '隐藏历史记录';
                    } else {
                        alert('请先开始录音创建会话');
                    }
                }
            }
            
            async function loadHistory() {
                try {
                    const response = await fetch(`/get_history?session_id=${sessionId}`);
                    const data = await response.json();
                    
                    // 清空历史容器
                    historyContainer.innerHTML = '<h2>对话历史记录</h2>';
                    
                    // 显示历史记录
                    if (data.history && data.history.length > 0) {
                        data.history.forEach((item, index) => {
                            const historyItem = document.createElement('div');
                            historyItem.className = 'history-item';
                            historyItem.innerHTML = `
                                <p><strong>时间:</strong> ${item.timestamp}</p>
                                <p><strong class="user-text">用户:</strong> ${item.user_input}</p>
                                <p><strong class="ai-text">AI:</strong> ${item.ai_response}</p>
                            `;
                            historyContainer.appendChild(historyItem);
                        });
                    } else {
                        historyContainer.innerHTML += '<p>暂无对话历史记录</p>';
                    }
                } catch (error) {
                    console.error('加载历史记录失败:', error);
                    historyContainer.innerHTML += '<p>加载历史记录失败</p>';
                }
            }
            
            async function startRecording() {
                try {
                    // 获取新的会话ID
                    const sessionResponse = await fetch('/start_session');
                    const sessionData = await sessionResponse.json();
                    sessionId = sessionData.session_id;
                    
                    // 音频上下文和流设置
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const source = audioContext.createMediaStreamSource(stream);
                    
                    // 创建新的录音处理器
                    const processorNode = audioContext.createScriptProcessor(4096, 1, 1);
                    source.connect(processorNode);
                    processorNode.connect(audioContext.destination);
                    
                    // 设置音频格式与采样率
                    const sampleRate = audioContext.sampleRate;
                    
                    // 处理音频数据
                    let audioChunks = [];
                    
                    processorNode.onaudioprocess = function(e) {
                        if (!isRecording) return;
                        
                        const buffer = e.inputBuffer.getChannelData(0);
                        
                        // 重采样到16kHz（如果需要）
                        let resampledBuffer;
                        if (sampleRate === 16000) {
                            resampledBuffer = buffer;
                        } else {
                            // 简单的降采样方法
                            const ratio = 16000 / sampleRate;
                            const newLength = Math.round(buffer.length * ratio);
                            resampledBuffer = new Float32Array(newLength);
                            
                            for (let i = 0; i < newLength; i++) {
                                const oldIndex = Math.floor(i / ratio);
                                resampledBuffer[i] = buffer[oldIndex];
                            }
                        }
                        
                        // 转换为16位PCM
                        const pcmBuffer = new Int16Array(resampledBuffer.length);
                        for (let i = 0; i < resampledBuffer.length; i++) {
                            const s = Math.max(-1, Math.min(1, resampledBuffer[i]));
                            pcmBuffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                        }
                        
                        audioChunks.push(pcmBuffer);
                    };
                    
                    // 存储录音器组件以便后续停止
                    recorder = {
                        processorNode: processorNode,
                        source: source,
                        stream: stream
                    };
                    
                    isRecording = true;
                    statusDiv.textContent = '持续录音中...系统会自动检测并处理你的语音';
                    statusDiv.className = 'recording';
                    recordButton.textContent = '停止录音';
                    
                    // 定期发送音频块到服务器（每200ms）
                    uploadInterval = setInterval(async () => {
                        if (audioChunks.length > 0) {
                            const wavData = createWavFile(audioChunks, 16000);
                            audioChunks = []; // 清空缓冲区，准备下一批
                            const response = await sendAudioChunk(wavData);
                            
                            // 检查是否需要打断正在播放的音频
                            if (response.status === 'interrupt' && activeAudioElement) {
                                console.log("检测到语音输入，打断当前音频播放");
                                activeAudioElement.pause();
                                activeAudioElement.currentTime = 0;
                                
                                // 通知服务器音频已停止播放
                                updateAudioStatus('stopped');
                            }
                            
                            // 检查是否有新的处理结果
                            if (response.status === 'new_result' && response.result) {
                                displayResult(response.result);
                            }
                        }
                    }, 200);
                    
                } catch (error) {
                    statusDiv.textContent = '无法访问麦克风: ' + error.message;
                }
            }
            
            // 更新音频播放状态
            async function updateAudioStatus(status) {
                if (!sessionId) return;
                
                const formData = new FormData();
                formData.append('session_id', sessionId);
                formData.append('status', status);
                
                try {
                    await fetch('/audio_status', {
                        method: 'POST',
                        body: formData
                    });
                } catch (error) {
                    console.error('更新音频状态失败:', error);
                }
            }
            
            // 创建WAV文件
            function createWavFile(audioChunks, sampleRate) {
                // 计算总数据长度
                let recordingLength = 0;
                for (let i = 0; i < audioChunks.length; i++) {
                    recordingLength += audioChunks[i].length;
                }
                
                // 创建最终数据缓冲区
                let offset = 0;
                const pcmBuffer = new Int16Array(recordingLength);
                for (let i = 0; i < audioChunks.length; i++) {
                    pcmBuffer.set(audioChunks[i], offset);
                    offset += audioChunks[i].length;
                }
                
                // 创建WAV头
                const WAV_HEADER_SIZE = 44;
                const wavBuffer = new ArrayBuffer(WAV_HEADER_SIZE + pcmBuffer.length * 2);
                const view = new DataView(wavBuffer);
                
                // RIFF标识
                writeString(view, 0, 'RIFF');
                view.setUint32(4, 36 + pcmBuffer.length * 2, true);
                writeString(view, 8, 'WAVE');
                
                // fmt子块
                writeString(view, 12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true); // PCM格式
                view.setUint16(22, 1, true); // 单声道
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true); // 字节率
                view.setUint16(32, 2, true); // 数据块对齐
                view.setUint16(34, 16, true); // 每个样本位数
                
                // 数据子块
                writeString(view, 36, 'data');
                view.setUint32(40, pcmBuffer.length * 2, true);
                
                // 写入PCM数据
                const pcmDataView = new DataView(wavBuffer, WAV_HEADER_SIZE);
                for (let i = 0; i < pcmBuffer.length; i++) {
                    pcmDataView.setInt16(i * 2, pcmBuffer[i], true);
                }
                
                return new Blob([wavBuffer], { type: 'audio/wav' });
            }
            
            // 辅助函数：写入字符串到DataView
            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }
            
            async function sendAudioChunk(wavBlob) {
                if (!wavBlob || wavBlob.size === 0) return { status: 'empty' };
                
                const formData = new FormData();
                formData.append('audio', wavBlob, 'chunk.wav');
                formData.append('session_id', sessionId);
                
                try {
                    const response = await fetch('/upload_chunk', {
                        method: 'POST',
                        body: formData
                    });
                    return await response.json();
                } catch (error) {
                    console.error('发送音频块失败:', error);
                    return { status: 'error', message: error.message };
                }
            }
            
            async function stopRecording() {
                if (!isRecording) return;
                
                isRecording = false;
                statusDiv.className = '';
                statusDiv.textContent = '录音已停止';
                recordButton.textContent = '开始录音';
                
                // 清除上传定时器
                clearInterval(uploadInterval);
                
                // 停止录音处理器
                if (recorder) {
                    recorder.processorNode.disconnect();
                    recorder.source.disconnect();
                    recorder.stream.getTracks().forEach(track => track.stop());
                    recorder = null;
                }
                
                // 关闭音频上下文
                if (audioContext && audioContext.state !== 'closed') {
                    await audioContext.close();
                }
                
                // 通知服务器会话结束（可选）
                const formData = new FormData();
                formData.append('session_id', sessionId);
                
                try {
                    await fetch('/end_session', {
                        method: 'POST',
                        body: formData
                    });
                } catch (error) {
                    console.error('结束会话失败:', error);
                }
            }
            
            function displayResult(result) {
                // 创建新的结果元素
                const resultElement = document.createElement('div');
                resultElement.className = 'result-item';
                
                resultElement.innerHTML = `
                    <p><strong>您说:</strong> ${result.input_text}</p>
                    <p><strong>AI回复:</strong> ${result.output_text}</p>
                    <audio controls src="${result.audio_url}" autoplay></audio>
                `;
                
                // 添加到结果容器的顶部
                resultContainer.insertBefore(resultElement, resultContainer.firstChild);
                
                // 获取新创建的音频元素并设置事件监听器
                const audioElement = resultElement.querySelector('audio');
                
                // 设置为当前活动的音频元素
                activeAudioElement = audioElement;
                
                // 当音频开始播放时通知服务器
                audioElement.addEventListener('play', function() {
                    updateAudioStatus('playing');
                });
                
                // 当音频停止或结束时通知服务器
                audioElement.addEventListener('pause', function() {
                    updateAudioStatus('stopped');
                });
                
                audioElement.addEventListener('ended', function() {
                    updateAudioStatus('stopped');
                    activeAudioElement = null;
                });
            }
        </script>
    </body>
    </html>
    '''

if __name__ == "__main__":
    # 加载模型
    print("加载模型中...")
    load_models()
    print("模型加载完成，启动服务器...")
    
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=7862)