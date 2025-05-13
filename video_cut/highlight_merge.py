import os
import requests
import json
import cv2
import numpy as np
from utils.s3_manager import S3Manager
import subprocess
import boto3
from utils.file_generator import generate_jpg_from_video
from moviepy.editor import VideoFileClip, concatenate_videoclips
from config import S3_LOCAL_DIR, RALLIES_JSON_URL_TEMPLATE, HIGHLIGHT_S3_PREFIX
from datetime import datetime

class HighlightMerger:
    def __init__(self, userid, timeslot, download_dir=S3_LOCAL_DIR):
        self.userid = userid
        self.timeslot = timeslot
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        # 音频临时目录直接放在S3-test根目录下
        self.audio_dir = os.path.join(self.download_dir, "audio_temp")
        os.makedirs(self.audio_dir, exist_ok=True)
        self.s3_manager = S3Manager()
        self.rallies_json_url = RALLIES_JSON_URL_TEMPLATE.format(userid=userid, timeslot=timeslot)
        self.rallies_data = self._load_rallies_json()

    def _load_rallies_json(self):
        resp = requests.get(self.rallies_json_url)
        if resp.status_code != 200:
            raise Exception(f"无法获取 rallies.json: {self.rallies_json_url}")
        return resp.json()

    def download_videos_by_ids(self, video_id_list):
        # 下载rallies.json文件
        print(f"正在获取 rallies.json，URL: {self.rallies_json_url}")
        response = requests.get(self.rallies_json_url)
        print(f"Response status: {response.status_code}")
        rallies_data = response.json()
        print(f"获取到的 rallies 数据: {rallies_data}")
        
        # 根据ID列表下载视频
        downloaded_files = []
        for video_id in video_id_list:
            print(f"\n处理视频 ID: {video_id}")
            # 在列表中查找匹配的 rally_id
            found_rally = None
            for rally in rallies_data:
                if rally['rally_id'] == video_id:
                    found_rally = rally
                    break
            
            if found_rally:
                # 从 videoUrl 中提取文件名
                video_url = found_rally['videoUrl']
                s3_key = f"output/{self.userid}/{self.timeslot}/rallies/{os.path.basename(video_url)}"
                local_path = os.path.join(self.download_dir, f"{video_id}.mp4")
                print(f"准备下载: S3 key={s3_key}, 本地路径={local_path}")
                try:
                    self.s3_manager.download_file(s3_key, local_path)
                    print(f"下载成功: {local_path}")
                    downloaded_files.append(local_path)
                except Exception as e:
                    print(f"下载失败: {str(e)}")
            else:
                print(f"警告: 视频 ID {video_id} 在 rallies 数据中未找到")
        
        print(f"\n总共下载文件数: {len(downloaded_files)}")
        print(f"下载的文件列表: {downloaded_files}")
        return downloaded_files

    def _get_video_url_by_id(self, video_id):
        for item in self.rallies_data:
            if item.get("rally_id") == video_id:
                return item.get("videoUrl")
        return None

    def _download_file(self, url, local_path):
        resp = requests.get(url, stream=True)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"已下载: {local_path}")
        else:
            print(f"下载失败: {url}")

    def merge_videos_with_audio(self, video_files, output_path):
        clips = [VideoFileClip(f) for f in video_files]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        for clip in clips:
            clip.close()

    def list_videos(self, prefix, bucket):
        """直接用 boto3 列举 S3 指定目录下所有 mp4 文件"""
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        video_files = []
        for page in page_iterator:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.mp4'):
                    video_files.append(key)
        return video_files

    def upload_and_index(self, video_path):
        # 视频文件命名：{userid}_{timeslot}.mp4
        video_filename = f"{self.userid}_{self.timeslot}.mp4"
        
        # 视频文件 S3 路径：highlights/{userid}/{timeslot}/{userid}_{timeslot}.mp4
        video_s3_key = f"highlights/{self.userid}/{self.timeslot}/{video_filename}"
        print(f"准备上传视频到 S3: {video_s3_key}")
        
        # 上传视频文件
        self.s3_manager.upload_file(video_path, video_s3_key)
        
        # 生成缩略图
        thumb_path = os.path.join(self.download_dir, f"{self.userid}_{self.timeslot}.jpg")
        generate_jpg_from_video(video_path, thumb_path)
        thumb_s3_key = f"highlights/{self.userid}/{self.timeslot}/{os.path.basename(thumb_path)}"
        self.s3_manager.upload_file(thumb_path, thumb_s3_key)
        
        # 索引文件 S3 路径：highlights/{userid}/{userid}_highlights.json
        index_key = f"highlights/{self.userid}/{self.userid}_highlights.json"
        
        # 读取现有索引文件（如果存在）
        try:
            existing_index = self.s3_manager.get_file(index_key)
            index_data = json.loads(existing_index)
        except:
            index_data = []
        
        # 添加新的视频信息
        video_info = {
            "userid": self.userid,
            "timeslot": self.timeslot,
            "video_url": f"https://chenqiwei.org/{video_s3_key}",
            "thumb_url": f"https://chenqiwei.org/{thumb_s3_key}",
            "filename": video_filename,
            "created_at": datetime.now().isoformat()
        }
        
        # 如果已存在相同 timeslot 的记录，更新它
        found = False
        for i, item in enumerate(index_data):
            if item.get("timeslot") == self.timeslot:
                index_data[i] = video_info
                found = True
                break
        
        # 如果不存在，添加新记录
        if not found:
            index_data.append(video_info)
        
        # 更新索引文件
        self.s3_manager.update_index(index_key, index_data)
        print(f"已更新索引文件: {index_key}")


