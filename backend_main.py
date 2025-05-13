import os
import logging
import cv2
from utils.s3_manager import S3Manager
from main import main
from utils import save_video
from utils.video_convert_utils import convert_to_h264, add_audio_to_video, extract_audio, ffmpeg_cut_video_with_audio
from utils.file_generator import generate_jpg_from_video, generate_json_for_rallies  # 新增导入

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_LOCAL_DIR = "S3-test"
os.makedirs(S3_LOCAL_DIR, exist_ok=True)

def process_video_from_s3(s3_key: str, output_prefix: str, video_id: str) -> dict:
    """
    从S3下载原始视频，调用main.py处理，并将处理后的视频上传回S3
    所有文件均放在S3-test目录下
    """
    s3_manager = S3Manager(bucket_name="rally-video-test1")
    
    # 下载原始视频到S3-test目录
    local_input_path = os.path.join(S3_LOCAL_DIR, os.path.basename(s3_key))
    logger.info(f"准备从S3下载视频: {s3_key} 到本地: {local_input_path}")
    s3_manager.download_video(s3_key, local_input_path)
    logger.info(f"已下载原始视频到: {local_input_path}")

 

    # 提取音频
    audio_path = os.path.join(S3_LOCAL_DIR, f"output_video{video_id}.mp3")
    extract_audio(local_input_path, audio_path)
    logger.info(f"已提取音频到: {audio_path}")

    # 调用main.py处理视频
    logger.info(f"开始处理本地视频: {local_input_path}")
    result = main(local_input_path)
    logger.info("视频分析处理完成")
    
    # 保存完整视频
    output_video_path = os.path.join(S3_LOCAL_DIR, f"output_video{video_id}.avi")
    save_video(result['output_video_frames'], output_video_path, fps=result['fps'])
    logger.info(f"已保存处理后视频到: {output_video_path}")
    
    # 转码完整视频为h264 mp4
    output_video_h264_path = os.path.join(S3_LOCAL_DIR, f"output_video{video_id}.mp4")
    convert_to_h264(output_video_path, output_video_h264_path)
    logger.info(f"已转码处理后视频到: {output_video_h264_path}")
    
    # 合成主视频音频（假设音频文件为output_video{video_id}.mp3）
    audio_path = os.path.join(S3_LOCAL_DIR, f"output_video{video_id}.mp3")
    if os.path.exists(audio_path):
        temp_video_with_audio = os.path.join(S3_LOCAL_DIR, f"output_video{video_id}_with_audio.mp4")
        add_audio_to_video(output_video_h264_path, audio_path, temp_video_with_audio)
        os.replace(temp_video_with_audio, output_video_h264_path)
        logger.info(f"已合成主视频音频: {output_video_h264_path}")
    else:
        logger.info(f"未找到主视频音频文件: {audio_path}，跳过音频合成")
    


    # 保存rallies视频
    rallies_dir = os.path.join(S3_LOCAL_DIR, "output_rallies")
    os.makedirs(rallies_dir, exist_ok=True)
    result['video_cutter'].save_rallies_as_videos(
        output_video_path,
        result['rallies'],
        rallies_dir
    )
    logger.info(f"已保存rallies视频到: {rallies_dir}")
    
    # rallies切分（用ffmpeg带音频切分）
    rallies_h264_dir = os.path.join(S3_LOCAL_DIR, "output_rallies_h264")
    os.makedirs(rallies_h264_dir, exist_ok=True)
    rally_urls = []
    fps = result['fps']
    for idx, rally in enumerate(result['rallies'], 1):
        dst_name = f"rally_{idx}.mp4"
        dst_path = os.path.join(rallies_h264_dir, dst_name)
        ffmpeg_cut_video_with_audio(
            output_video_h264_path, dst_path,
            rally['start_frame'], rally['end_frame'], fps
        )
        logger.info(f"已ffmpeg切分rally视频: {dst_path}")
        # 上传rally视频
        rally_key = f"{output_prefix}/rallies/{dst_name}"
        rally_url = s3_manager.upload_video(dst_path, rally_key)
        rally_urls.append(rally_url)
        logger.info(f"已上传rally视频到: {rally_url}")

        # 生成并上传rally截图
        rally_jpg_name = f"rally_{idx}.jpg"
        rally_jpg_path = os.path.join(rallies_h264_dir, rally_jpg_name)
        generate_jpg_from_video(dst_path, rally_jpg_path)
        rally_jpg_key = f"{output_prefix}/rallies/{rally_jpg_name}"
        s3_manager.upload_video(rally_jpg_path, rally_jpg_key)
        logger.info(f"已上传rally截图到: {rally_jpg_key}")

    # 生成rallies目录下的索引JSON并上传
    rallies_json_path = os.path.join(rallies_h264_dir, "rallies.json")
    rallies_base_url = f"{output_prefix}/rallies"
    generate_json_for_rallies(rallies_h264_dir, rallies_json_path, rallies_base_url)
    rallies_json_key = f"{output_prefix}/rallies/rallies.json"
    s3_manager.upload_video(rallies_json_path, rallies_json_key)
    logger.info(f"已上传rallies目录索引JSON到: {rallies_json_key}")

    # 上传处理后的视频
    output_video_key = f"{output_prefix}/{video_id}-{timeslot}.mp4"
    output_video_url = s3_manager.upload_video(output_video_h264_path, output_video_key)
    logger.info(f"已上传处理后视频到: {output_video_url}")
    
    return {
        "output_video_url": output_video_url,
        "rally_urls": rally_urls
    }

if __name__ == "__main__":
    # 示例调用
    user_id = "user1"
    video_id = "match1"
    timeslot = "10-00-11-00"
    input_key = f"input/{user_id}/{timeslot}/{video_id}-{timeslot}.mp4"
    output_prefix = f"output/{user_id}/{timeslot}"
    result = process_video_from_s3(input_key, output_prefix, video_id)
    print(result)