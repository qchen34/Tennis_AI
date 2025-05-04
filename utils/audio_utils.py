import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

class AudioProcessor:
    def __init__(self, sr=22050, frame_duration=0.02, energy_threshold=0.5, zcr_threshold=0.1):
        """
        初始化音频处理器。
        :param sr: 采样率。
        :param frame_duration: 每帧的持续时间（秒）。
        :param energy_threshold: 短时能量的阈值。
        :param zcr_threshold: 零交叉率的阈值。
        """
        self.sr = sr
        self.frame_duration = frame_duration
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold

    def extract_audio_signal(self, video_path):
        """
        从视频中提取音频信号。
        :param video_path: 视频文件路径。
        :return: 音频信号和采样率。
        """
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_array = audio.to_soundarray(fps=self.sr)
        mono_audio = np.mean(audio_array, axis=1)  # 转换为单声道
        return mono_audio, self.sr

    def detect_ball_hit_sounds(self, video_path):
        """
        检测音频中的击球声音。
        :param video_path: 视频文件路径。
        :return: 击球声音的时间点（秒）。
        """
        # 提取音频信号
        y, sr = self.extract_audio_signal(video_path)

        # 计算每帧的样本数
        frame_length = int(self.sr * self.frame_duration)
        hop_length = frame_length // 2

        # 计算短时能量和零交叉率
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]

        # 检测击球声音
        hit_frames = []
        for i, (e, z) in enumerate(zip(energy, zcr)):
            if e > self.energy_threshold and z > self.zcr_threshold:
                hit_frames.append(i)

        # 转换为时间点（秒）
        hop_duration = hop_length / self.sr
        hit_times = [frame * hop_duration for frame in hit_frames]
        return hit_times