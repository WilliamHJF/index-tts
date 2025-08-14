# pip install index-tts
from indextts.infer import IndexTTS
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

def preprocess_audio(audio_path, target_sr=24000, max_duration=10.0):
    """Preprocess audio to ensure consistent format"""
    try:
        # 使用librosa加载，它能更好地处理各种格式
        audio, sr = librosa.load(audio_path, sr=None)

        # 如果采样率不是16kHz，则重采样
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        # 如果是立体声，则转为单声道
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        # 计算当前时长
        duration = len(audio) / sr
        print(f"Original audio duration: {duration:.2f}s")

        # 如果时长超过8秒，则选择最佳片段
        if duration > max_duration:
            max_samples = int(max_duration * sr)
            audio = select_best_segment_by_energy(audio, max_samples, sr)
        duration = len(audio) / sr
        print(f"Updated audio duration: {duration:.2f}s")

        # 归一化，将振幅调整到[-0.9, 0.9]之间
        audio = audio / np.max(np.abs(audio)) * 0.9

        return audio, target_sr
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

def select_best_segment_by_energy(audio, max_samples, sr, window_duration=0.5):
    """
    基于能量密度选择最佳音频片段
    """
    if len(audio) <= max_samples:
        return audio

    # 计算滑动窗口的能量
    window_samples = int(window_duration * sr)
    step_size = window_samples // 4  # 25% 重叠

    energies = []
    positions = []
    for start in range(0, len(audio) - max_samples + 1, step_size):
        segment = audio[start:start + max_samples]

        # 计算RMS能量，忽略静音部分
        rms_energy = np.sqrt(np.mean(segment ** 2))
        # 计算过零率（语音活动指标）
        zero_crossings = np.sum(np.abs(np.diff(np.sign(segment)))) / len(segment)
        # 综合评分：能量 + 语音活动度
        score = rms_energy * (1 + zero_crossings * 0.1)

        energies.append(score)
        positions.append(start)

    # 选择能量最高的片段
    best_idx = np.argmax(energies)
    best_start = positions[best_idx]

    return audio[best_start:best_start + max_samples]

def enhance_audio_quality(audio_path, target_sr):
    """Apply basic audio enhancement techniques"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)

        # 如果采样率不是16kHz，则重采样
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        # 如果是立体声，则转为单声道
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        # 归一化，将振幅调整到[-0.9, 0.9]之间
        audio = audio / np.max(np.abs(audio)) * 0.9
        # 应用预加重滤波器，提升语音清晰度
        audio_filtered = librosa.effects.preemphasis(audio, coef=0.95)
        # 再次归一化，确保音量均衡
        audio_normalized = librosa.util.normalize(audio_filtered) * 0.95

        return audio_normalized
    except Exception as e:
        print(f"Error enhancing audio: {e}")
        return None

def analyze_audio_quality(audio_path):
    """简单的音频质量分析"""
    audio, sr = librosa.load(audio_path, sr=None)
    # 计算信噪比估计
    energy = np.mean(audio ** 2)
    # 计算频谱丰富度
    stft = librosa.stft(audio)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=np.abs(stft)))
    # 综合质量评分（简化版）
    quality_score = np.log10(energy + 1e-8) + spectral_centroid / 1000

    return quality_score

# 音频质量分析
def compare_audio_quality(audio_path1, audio_path2):
    """比较音频质量"""
    quality1 = analyze_audio_quality(audio_path1)
    quality2 = analyze_audio_quality(audio_path2)
    print(f"Analyzing audio quality1: {quality1}")
    print(f"Analyzing audio quality2: {quality2}")

if __name__ == '__main__':
    os.makedirs("result/", exist_ok=True)
    os.makedirs("temp/", exist_ok=True)
    task = pd.read_csv("aigc_speech_generation_tasks/aigc_speech_generation_tasks.csv")
    error_input = []

    # Check if we can use Index-TTS model
    try:
        tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True,
                       use_cuda_kernel=False)
        print("IndexTTS model loaded successfully")
    except Exception as e:
        print(f"IndexTTS model not available: {e}")
        raise

    # 测试用例
    row = 17
    try:
        new_audio, target_sr = preprocess_audio(f"./aigc_speech_generation_tasks/reference_{row}.wav")
        sf.write(f"temp/reference_{row}.wav", new_audio, target_sr)
        tts.infer(audio_prompt=f"./temp/reference_{row}.wav",
                  text="植物病害的发生往往与病原微生物的侵染有关，例如真菌、细菌、病毒等都可能引起植物组织病变，影响其正常生长和产量，因此在农业生产中需加强病害监测与防治。",
                  output_path=f"result/{row}.wav", verbose=True)
        enhanced_audio = enhance_audio_quality(f"result/{row}.wav", target_sr)
        if enhanced_audio is not None:
            sf.write(f"result/{row}.wav", enhanced_audio, target_sr)
    except Exception as e:
        print(f'用例输出错误: {e}')
        raise

    # 批量输出
    for row in task.iterrows():
        # index-tts 模型
        # success = False
        try:
            # Method1: 对所有音频进行优先预处理
            new_audio, target_sr = preprocess_audio(f"./aigc_speech_generation_tasks/{row[1].reference_speech}",
                                                    max_duration=10.0)
            sf.write(f"./temp/{row[1].reference_speech}", new_audio, target_sr)
            # 音频生成
            tts.infer(audio_prompt=f"./temp/{row[1].reference_speech}",
                      text=f"{row[1].text}",
                      output_path="result/" + str(row[1].utt) + ".wav", verbose=True)
            # success = True
            # 音频增强
            enhanced_audio = enhance_audio_quality("result/" + str(row[1].utt) + ".wav", target_sr)
            if enhanced_audio is not None:
                sf.write("result/" + str(row[1].utt) + ".wav", enhanced_audio, target_sr)
        except Exception as e:
            error_input.append(row[1].utt)
            print("出错了")

        # Method2：仅针对模型报错的过长音频进行预处理
        # if not success:
        #     try:
        #         audio, target_sr = preprocess_audio(f"./aigc_speech_generation_tasks/{row[1].reference_speech}")
        #         sf.write(f"./temp/{row[1].reference_speech}", audio, target_sr)
        #         tts.infer(audio_prompt=f"./temp/{row[1].reference_speech}",
        #                   text=f"{row[1].text}",
        #                   output_path="result/" + str(row[1].utt) + ".wav", verbose=True)
        #         success = True
        #     except Exception as e:
        #         error_input.append(row[1].utt)
        #         print("还是出错了")

    print(error_input)
    task['synthesized_speech'] = [str(i) + ".wav" for i in range(1, 201)]
    task.to_csv("result/result.csv", index=None)