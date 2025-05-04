from flask import Flask, request, jsonify
from typing import Dict, Any, List, Tuple
import os
import tempfile
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from .storage import GCSManager
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
from mini_court.mini_court import MiniCourt
from video_cut.video_cut import VideoCut
from utils.compute_player_stats_utils import compute_player_stats
import cv2
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
gcs_manager = GCSManager()
executor = ThreadPoolExecutor(max_workers=4)

# 任务状态存储
task_status = {}

def process_video(video_path: str, task_id: str) -> Dict[str, Any]:
    """处理视频的核心函数
    
    Args:
        video_path: 视频文件路径
        task_id: 任务ID
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    try:
        # 更新任务状态
        task_status[task_id] = {'status': 'processing', 'progress': 0}
        
        # 读取视频
        video_frames, fps = read_video(video_path)
        task_status[task_id]['progress'] = 10
        
        # 初始化检测器
        player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
        ball_tracker = BallTracker(model_path='models/yolo9_best_1.pt')
        court_line_detector = CourtLineDetector("models/keypoints_model.pth")
        task_status[task_id]['progress'] = 20
        
        # 检测球员和球
        player_detections = player_tracker.detect_frames(video_frames)
        ball_detections = ball_tracker.detect_frames(video_frames)
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
        task_status[task_id]['progress'] = 40
        
        # 检测球场关键点
        court_keypoints = court_line_detector.predict(video_frames[0])
        task_status[task_id]['progress'] = 50
        
        # 过滤球员
        player_detections_filtered = player_tracker.choose_and_filter_players(
            court_keypoints, player_detections)
        
        # 初始化迷你球场
        mini_court = MiniCourt(video_frames[0])
        
        # 检测击球帧
        ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
        task_status[task_id]['progress'] = 60
        
        # 转换坐标
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections_filtered, ball_detections, court_keypoints)
        task_status[task_id]['progress'] = 70
        
        # 计算统计数据
        player_stats_data_df = compute_player_stats(
            ball_shot_frames,
            ball_mini_court_detections,
            player_mini_court_detections,
            mini_court,
            len(video_frames),
            fps=fps
        )
        task_status[task_id]['progress'] = 80
        
        # 处理视频片段
        video_cutter = VideoCut(
            ball_detections=ball_detections,
            ball_shot_frames=ball_shot_frames,
            acceleration_threshold=50.0,
            frame_merge_threshold=10
        )
        
        refined_hit_points = video_cutter.refine_hit_points()
        rallies = video_cutter.generate_rallies(
            refined_hit_points, max_frame_gap=240, buffer_frames=120)
        task_status[task_id]['progress'] = 90
        
        # 保存处理后的视频
        output_video_path = os.path.join(tempfile.gettempdir(), f'output_{task_id}.mp4')
        save_video(video_frames, output_video_path)
        
        # 上传处理后的视频
        output_url = gcs_manager.upload_video(
            output_video_path,
            f'processed/{task_id}.mp4'
        )
        
        # 清理临时文件
        os.unlink(video_path)
        os.unlink(output_video_path)
        
        task_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'result': {
                'output_video_url': output_url,
                'player_stats': player_stats_data_df.to_dict(),
                'rallies': rallies
            }
        }
        
        return task_status[task_id]['result']
        
    except Exception as e:
        logger.error(f"处理视频失败: {str(e)}")
        task_status[task_id] = {
            'status': 'failed',
            'error': str(e)
        }
        raise

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """处理视频分析请求"""
    try:
        # 获取视频 URL
        video_url = request.json.get('video_url')
        if not video_url:
            return jsonify({'error': 'Missing video_url'}), 400
            
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 创建临时文件
        temp_path = os.path.join(tempfile.gettempdir(), f'input_{task_id}.mp4')
        
        # 下载视频
        gcs_manager.download_video(video_url, temp_path)
        
        # 异步处理视频
        executor.submit(process_video, temp_path, task_id)
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '视频处理任务已开始'
        })
        
    except Exception as e:
        logger.error(f"处理请求失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def read_video(video_path: str) -> Tuple[np.ndarray, int]:
    """读取视频文件"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return np.array(frames), fps

def save_video(frames: np.ndarray, output_path: str) -> None:
    """保存视频文件"""
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release() 