import subprocess
import os

def convert_to_h264(input_path: str, output_path: str, crf: int = 23, preset: str = 'medium'):
    """
    使用ffmpeg将输入视频转码为h264编码的mp4格式。
    :param input_path: 输入视频文件路径
    :param output_path: 输出mp4文件路径
    :param crf: 质量参数，越小质量越高，默认23
    :param preset: 编码速度与压缩率平衡，默认'medium'
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"视频转码失败: {e.stderr.decode()}")
        raise

def add_audio_to_video(video_path: str, audio_path: str, output_path: str):
    """
    使用ffmpeg将音频合成到视频中，输出带音频的视频。
    :param video_path: 输入视频文件路径
    :param audio_path: 输入音频文件路径
    :param output_path: 输出合成后的视频路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"音频合成失败: {e.stderr.decode()}")
        raise

def extract_audio(video_path: str, audio_path: str):
    """
    使用ffmpeg从视频中提取音频，保存为mp3文件。
    :param video_path: 输入视频文件路径
    :param audio_path: 输出音频文件路径（mp3）
    """
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'mp3', audio_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"音频提取失败: {e.stderr.decode()}")
        raise

def ffmpeg_cut_video_with_audio(input_path, output_path, start_frame, end_frame, fps):
    """
    使用ffmpeg按帧号切分带音频的视频片段。
    :param input_path: 输入视频文件路径（需带音频）
    :param output_path: 输出切片视频路径
    :param start_frame: 起始帧号
    :param end_frame: 结束帧号
    :param fps: 视频帧率
    """
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', input_path,
        '-t', str(duration),
        '-c:v', 'copy',
        '-c:a', 'copy',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f'ffmpeg切片失败: {e.stderr.decode()}')
        raise
