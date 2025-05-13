# 导入自定义的S3视频下载类
from video_cut.highlight_merge import HighlightMerger
# 导入生成视频截图和json索引的工具函数
from utils.file_generator import generate_jpg_from_video, generate_json_for_rallies
import os
from utils.s3_manager import S3Manager
import argparse
from config import S3_LOCAL_DIR
import requests

if __name__ == "__main__":
    # 用户ID
    # userid = "user1"
    # 时间段
    # timeslot = "10-00-11-00"
    # 需要下载和处理的视频ID列表
    # video_id_list = ["rally_1", "rally_2", "rally_3"]
    parser = argparse.ArgumentParser(description='Process video highlights.')
    parser.add_argument('--userid', required=True, help='User ID')
    parser.add_argument('--timeslot', required=True, help='Time slot')
    parser.add_argument('--video_ids', required=True, help='Comma-separated list of video IDs')
    args = parser.parse_args()

    userid = args.userid
    timeslot = args.timeslot
    video_id_list = args.video_ids.split(',')

    # 初始化HighlightMerger，指定下载目录为S3-test根目录
    merger = HighlightMerger(userid, timeslot, download_dir=S3_LOCAL_DIR)
    # 根据ID列表下载视频，返回本地文件路径列表
    downloaded_files = merger.download_videos_by_ids(video_id_list)
    # 合成视频的输出路径和名称
    video_name = f"highlight_{userid}_{timeslot}.mp4"
    merged_video_path = os.path.join(S3_LOCAL_DIR, video_name)
    # 合成视频（简单拼接，含音频）
    merger.merge_videos_with_audio(downloaded_files, merged_video_path)
    # 上传合成视频并更新索引
    merger.upload_and_index(merged_video_path)


