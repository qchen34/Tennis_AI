from google.cloud import storage
import os
import logging
from typing import Optional, List, Dict, Any
from google.api_core.exceptions import GoogleAPIError

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSManager:
    def __init__(self, bucket_name: str = "tennis-analysis-videos"):
        """初始化 GCS 管理器
        
        Args:
            bucket_name: GCS 存储桶名称
        """
        try:
            self.storage_client = storage.Client()
            self.bucket_name = bucket_name
            self.bucket = self.storage_client.bucket(bucket_name)
            logger.info(f"成功初始化 GCS 管理器，使用存储桶: {bucket_name}")
        except GoogleAPIError as e:
            logger.error(f"初始化 GCS 管理器失败: {str(e)}")
            raise
    
    def upload_video(self, file_path: str, destination_blob_name: str) -> str:
        """上传视频到 GCS
        
        Args:
            file_path: 本地视频文件路径
            destination_blob_name: GCS 中的目标路径
            
        Returns:
            str: 上传后的公开访问 URL
            
        Raises:
            GoogleAPIError: 上传失败时抛出
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
                
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(file_path)
            blob.make_public()
            logger.info(f"成功上传视频到: {destination_blob_name}")
            return blob.public_url
        except (GoogleAPIError, FileNotFoundError) as e:
            logger.error(f"上传视频失败: {str(e)}")
            raise
    
    def download_video(self, blob_name: str, destination_path: str) -> None:
        """从 GCS 下载视频
        
        Args:
            blob_name: GCS 中的文件路径
            destination_path: 本地保存路径
            
        Raises:
            GoogleAPIError: 下载失败时抛出
        """
        try:
            blob = self.bucket.blob(blob_name)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            blob.download_to_filename(destination_path)
            logger.info(f"成功下载视频到: {destination_path}")
        except GoogleAPIError as e:
            logger.error(f"下载视频失败: {str(e)}")
            raise
    
    def delete_video(self, blob_name: str) -> None:
        """删除 GCS 中的视频
        
        Args:
            blob_name: GCS 中的文件路径
            
        Raises:
            GoogleAPIError: 删除失败时抛出
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.info(f"成功删除视频: {blob_name}")
        except GoogleAPIError as e:
            logger.error(f"删除视频失败: {str(e)}")
            raise
    
    def list_videos(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出 GCS 中的视频
        
        Args:
            prefix: 文件前缀过滤
            
        Returns:
            List[Dict[str, Any]]: 视频列表，每个视频包含名称、大小和更新时间
            
        Raises:
            GoogleAPIError: 列出文件失败时抛出
        """
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            videos = []
            for blob in blobs:
                videos.append({
                    'name': blob.name,
                    'size': blob.size,
                    'updated': blob.updated,
                    'url': blob.public_url
                })
            logger.info(f"成功列出视频，共 {len(videos)} 个")
            return videos
        except GoogleAPIError as e:
            logger.error(f"列出视频失败: {str(e)}")
            raise 