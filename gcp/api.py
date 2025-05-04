from flask import Flask, request, jsonify
from .functions import app, task_status
from .storage import GCSManager
import os
import logging
from typing import Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gcs_manager = GCSManager()

@app.route('/upload', methods=['POST'])
def upload_video():
    """处理视频上传请求"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if not video_file.filename:
            return jsonify({'error': 'No file selected'}), 400

        # 创建临时文件
        temp_path = os.path.join(os.getcwd(), 'temp', video_file.filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        video_file.save(temp_path)

        # 上传到 GCS
        blob_name = f'videos/{video_file.filename}'
        video_url = gcs_manager.upload_video(temp_path, blob_name)

        # 清理临时文件
        os.remove(temp_path)

        logger.info(f"成功上传视频: {video_file.filename}")
        return jsonify({
            'success': True,
            'video_url': video_url
        })

    except Exception as e:
        logger.error(f"上传视频失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_job_status(task_id: str):
    """获取任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        Dict[str, Any]: 任务状态信息
    """
    try:
        if task_id not in task_status:
            return jsonify({
                'error': f'Task {task_id} not found'
            }), 404
            
        status = task_status[task_id]
        logger.info(f"查询任务状态: {task_id} - {status['status']}")
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"查询任务状态失败: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/results/<task_id>', methods=['GET'])
def get_job_results(task_id: str):
    """获取任务结果
    
    Args:
        task_id: 任务ID
        
    Returns:
        Dict[str, Any]: 任务结果
    """
    try:
        if task_id not in task_status:
            return jsonify({
                'error': f'Task {task_id} not found'
            }), 404
            
        status = task_status[task_id]
        if status['status'] != 'completed':
            return jsonify({
                'error': f'Task {task_id} is not completed yet'
            }), 400
            
        logger.info(f"获取任务结果: {task_id}")
        return jsonify(status['result'])
        
    except Exception as e:
        logger.error(f"获取任务结果失败: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500 