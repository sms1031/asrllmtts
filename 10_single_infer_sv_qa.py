#http->https
#ssh -R 80:localhost:7861 serveo.net

import flask
from flask import Flask, request, jsonify
#import wave
import threading
import numpy as np
import time
#import webrtcvad
import os
import threading
#from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# --- 配置huggingFace国内镜像 ---
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 参数设置
AUDIO_RATE = 16000        # 音频采样率
AUDIO_CHANNELS = 1        # 单声道
VAD_MODE = 3              # VAD 模式 (0-3, 数字越大越敏感)
OUTPUT_DIR = "./output"   # 输出目录
#folder_path = "./Test_QWen2_VL/"
audio_file_count = 0

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
#os.makedirs(folder_path, exist_ok=True)

# 初始化 WebRTC VAD
#vad = webrtcvad.Vad()
#vad.set_mode(VAD_MODE)

# 创建Flask应用
app = Flask(__name__)

# 初始化模型（在全局范围内加载一次）
def load_models():
    global model_senceVoice
    
    # -------- SenceVoice 语音识别 --模型加载-----
    from funasr import AutoModel
    model_dir = r"iic/SenseVoiceSmall"
    model_senceVoice = AutoModel(model=model_dir, 
    				  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                      trust_remote_code=True)

    return "Models loaded successfully"

# 音频处理API
@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # 保存上传的音频文件
    global audio_file_count
    audio_file_count += 1
    filename = f"audio_{audio_file_count}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    audio_file.save(filepath)
    
    # 处理音频并获取回复
    result = process_audio_file(filepath)
    
    return jsonify(result)

# 处理音频文件
def process_audio_file(audio_path):
    # -------- SenceVoice 推理 ---------
    res = model_senceVoice.generate(
        input=audio_path,
        cache={},
        language="auto",
        use_itn=False,
    )
    prompt = res[0]['text'].split(">")[-1]
    print("ASR OUT:", prompt)
    
    # 返回处理结果
    return {
        'input_text': prompt
    }

# 主页
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
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background: #45a049; }
            #status, #result { margin-top: 20px; }
            .recording { color: red; }
        </style>
    </head>
    <body>
        <h1>语音助手服务</h1>
        <p>点击下方按钮开始录音，与AI助手对话</p>
        
        <button id="recordButton">开始录音</button>
        <div id="status"></div>
        <div id="result"></div>
        
        <script>
            const recordButton = document.getElementById('recordButton');
            const statusDiv = document.getElementById('status');
            const resultDiv = document.getElementById('result');
            
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            
            recordButton.addEventListener('click', toggleRecording);
            
            function toggleRecording() {
                if (isRecording) {
                    stopRecording();
                } else {
                    startRecording();
                }
                isRecording = !isRecording;
            }
            
            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.addEventListener('dataavailable', event => {
                        audioChunks.push(event.data);
                    });
                    
                    mediaRecorder.addEventListener('stop', async () => {
                        statusDiv.textContent = '处理中...';
                        recordButton.textContent = '开始录音';
                        
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const formData = new FormData();
                        formData.append('audio', audioBlob);
                        
                        try {
                            const response = await fetch('/process_audio', {
                                method: 'POST',
                                body: formData
                            });
                            
                            const result = await response.json();
                            displayResult(result);
                        } catch (error) {
                            statusDiv.textContent = '处理失败: ' + error.message;
                        }
                    });
                    
                    mediaRecorder.start();
                    statusDiv.textContent = '录音中...';
                    statusDiv.className = 'recording';
                    recordButton.textContent = '停止录音';
                } catch (error) {
                    statusDiv.textContent = '无法访问麦克风: ' + error.message;
                }
            }
            
            function stopRecording() {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                    statusDiv.className = '';
                }
            }
            
            function displayResult(result) {
                statusDiv.textContent = '处理完成';
                
                const html = `
                    <div>
                        <p><strong>您说:</strong> ${result.input_text}</p>
                    </div>
                `;
                
                resultDiv.innerHTML = html;
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
    app.run(host='0.0.0.0', port=7861)
    #app.run(host='0.0.0.0', port=7861, ssl_context=('cert.pem', 'key.pem'))
