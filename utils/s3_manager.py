import boto3
import os
import logging
import json
from typing import Optional, List, Dict, Any
from botocore.exceptions import ClientError
from config import S3_BUCKET, S3_LOCAL_DIR

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self, bucket_name=S3_BUCKET):
        """初始化 S3 管理器
        
        Args:
            bucket_name: S3 存储桶名称
        """
        try:
            self.s3 = boto3.client('s3')
            self.bucket_name = bucket_name
            self.local_dir = S3_LOCAL_DIR
            logger.info(f"成功初始化 S3 管理器，使用存储桶: {bucket_name}")
        except ClientError as e:
            logger.error(f"初始化 S3 管理器失败: {str(e)}")
            raise
    
    def upload_video(self, file_path: str, destination_key: str) -> str:
        """上传视频到 S3
        
        Args:
            file_path: 本地视频文件路径
            destination_key: S3 中的目标路径
            
        Returns:
            str: 上传后的公开访问 URL
            
        Raises:
            ClientError: 上传失败时抛出
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            self.s3.upload_file(file_path, self.bucket_name, destination_key)
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{destination_key}"
            logger.info(f"成功上传视频到: {destination_key}")
            return url
        except (ClientError, FileNotFoundError) as e:
            logger.error(f"上传视频失败: {str(e)}")
            raise
    
    def download_video(self, key: str, destination_path: str) -> None:
        """从 S3 下载视频
        
        Args:
            key: S3 中的文件路径
            destination_path: 本地保存路径
            
        Raises:
            ClientError: 下载失败时抛出
        """
        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            self.s3.download_file(self.bucket_name, key, destination_path)
            logger.info(f"成功下载视频到: {destination_path}")
        except ClientError as e:
            logger.error(f"下载视频失败: {str(e)}")
            raise
    
    def delete_video(self, key: str) -> None:
        """删除 S3 中的视频
        
        Args:
            key: S3 中的文件路径
            
        Raises:
            ClientError: 删除失败时抛出
        """
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"成功删除视频: {key}")
        except ClientError as e:
            logger.error(f"删除视频失败: {str(e)}")
            raise
    
    def list_videos(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出 S3 中的视频
        
        Args:
            prefix: 文件前缀过滤
            
        Returns:
            List[Dict[str, Any]]: 视频列表，每个视频包含名称、大小和更新时间
            
        Raises:
            ClientError: 列出文件失败时抛出
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            videos = []
            for obj in response.get('Contents', []):
                videos.append({
                    'name': obj['Key'],
                    'size': obj['Size'],
                    'updated': obj['LastModified'],
                    'url': f"https://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}"
                })
            logger.info(f"成功列出视频，共 {len(videos)} 个")
            return videos
        except ClientError as e:
            logger.error(f"列出视频失败: {str(e)}")
            raise

    def download_file(self, s3_key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket_name, s3_key, local_path)
        return local_path

    def upload_file(self, local_path, s3_key):
        self.s3.upload_file(local_path, self.bucket_name, s3_key)
        return f"s3://{self.bucket_name}/{s3_key}"

    def list_files(self, prefix):
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]

    def get_file_url(self, s3_key):
        return f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"

    def update_index(self, index_key, data):
        self.s3.put_object(Bucket=self.bucket_name, Key=index_key, Body=json.dumps(data)) 