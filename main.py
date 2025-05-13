from utils import (read_video, 
                   save_video,
                   draw_player_stats,
                   compute_player_stats
                   )
import constants
import pandas as pd
import numpy as np
from copy import deepcopy
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from video_cut import VideoCut
from utils.audio_utils import AudioProcessor
import matplotlib.pyplot as plt
import librosa.display
from utils.video_convert_utils import convert_to_h264


def main(input_video_path: str = "input/tennis_match_video1.mp4"):
    print("开始执行main函数...")
    
    # Read Video
    print("1. 开始读取视频...")
    video_frames, fps = read_video(input_video_path)
    print(f"视频读取完成，共{len(video_frames)}帧")

    # Detect Players and Ball
    print("2. 初始化球员跟踪器...")
    player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
    print("3. 初始化球跟踪器...")
    ball_tracker = BallTracker(model_path='models/yolo9_best_1.pt')
    
    print("4. 开始检测球员...")
    player_detections = player_tracker.detect_frames(video_frames, 
                                                    read_from_stub=True, 
                                                    stub_path="tracker_stubs/player_detections.pkl")
    print(f"球员检测完成，检测到{len(player_detections)}帧的球员")

    # 打印每帧中检测到的球员 ID
    # print("检测到的球员 ID:")
    # for frame_idx, player_dict in enumerate(player_detections):
    #     player_ids = list(player_dict.keys())
    #     print(f"Frame {frame_idx}: Detected player IDs: {player_ids}")


    print("5. 开始检测球...")
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                read_from_stub=True, 
                                                stub_path="tracker_stubs/ball_detections.pkl")
    print(f"球检测完成，检测到{len(ball_detections)}帧的球")
    
    print("6. 开始插值球的位置...")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    print("球位置插值完成")

    # Court Line Detector model
    print("7. 初始化球场检测器...")
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    print("8. 开始检测球场关键点...")
    court_keypoints = court_line_detector.predict(video_frames[0])
    print(f"球场关键点检测完成，检测到{len(court_keypoints)}个关键点")

    print("Court keypoints:", court_keypoints)

    print("9. 开始选择和过滤球员...")
    player_detections_filtered = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    print(f"球员选择和过滤完成，剩余{len(player_detections)}帧的球员数据")

    # 调用 match_players 函数
    # tracked_players = player_tracker.match_players(video_frames[0], player_detections[0])
    
    # 打印每个球员 ID 和绑定的颜色特征
    # print("\n检测到的球员 ID 和绑定颜色特征:")
    # for player_id, bbox in tracked_players:
    #     hist = player_tracker.players[player_id]  # 获取绑定的颜色直方图
    #     print(f"Player ID: {player_id}, BBox: {bbox}, Color Histogram (first 5 bins): {hist.flatten()[:5]}")

    #打印每帧中检测到的球员 ID
    # print("检测到的球员 ID:")
    # for frame_idx, player_dict in enumerate(player_detections_filtered):
    #    player_ids = list(player_dict.keys())
    #    print(f"Frame {frame_idx}: Detected player IDs: {player_ids}")
    

    # Mini Court
    print("10. 初始化迷你球场...")
    mini_court = MiniCourt(video_frames[0]) 

    # Detect ball shots
    print("11. 开始检测击球帧...")
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(f"击球帧检测完成: {ball_shot_frames}")

    # 打印 player_detections 的前几个数据
    # print("\nPlayer detections (first 5 frames):")
    # for i, detection in enumerate(player_detections_filtered[:5]):
    #     print(f"Frame {i}: {detection}")

    # 打印 player_detections 的前几个数据
    # print("\nPlayer detections (first 5 frames):")
    # for i, detection in enumerate(player_detections[:5]):
    #     print(f"Frame {i}: {detection}")

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections_filtered, ball_detections, court_keypoints)

    # 计算球员统计数据
    player_stats_data_df = compute_player_stats(
        ball_shot_frames, 
        ball_mini_court_detections, 
        player_mini_court_detections, 
        mini_court,
        len(video_frames),
        fps = fps
    )
    
    # Draw output
    print("12. 开始绘制输出...")
    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections_filtered)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
       cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    # Save video
    print("13. 开始保存视频...")
    save_video(output_video_frames, "output_videos/output_video.avi", fps=fps)
    print("14. 所有处理完成！")

    """
    # 打印 player_detections 的前几个数据
    print("\nPlayer detections (first 5 frames):")
    for i, detection in enumerate(player_detections[:5]):
        print(f"Frame {i}: {detection}")
    
    print("\nBall detections (first 5 frames):")
    for i, detection in enumerate(ball_detections[:5]):
        print(f"Frame {i}: {detection}")
    """


    # 初始化 VideoCut 类
    video_cutter = VideoCut(
        ball_detections=ball_detections,
        ball_shot_frames=ball_shot_frames,
        acceleration_threshold=50.0,  # 加速度阈值
        frame_merge_threshold=10      # 帧数合并阈值
    )

    # 获取优化后的击球点
    """
    当前视频实际的击球点，人工判断如下：
    [33, 88, 240, 322, 461, 540, 695, 900, 959, 1011, 1129, 1150, 1400, 1448, 1510, 1600, 1700, 1780, 3295, 3350, 3459, 3538]
    """
    # refined_hit_points = video_cutter.refine_hit_points()
    refined_hit_points = [33, 88, 240, 322, 461, 540, 695, 900, 959, 1011, 1129, 1150, 1400, 1448, 1510, 1600, 1700, 1780, 3295, 3350, 3459, 3538]
    print("Refined hit points (frames):", refined_hit_points)

    # 生成rallies
    rallies = [{'start_frame': 0, 'end_frame': 800, 'hit_points': [33, 88, 240, 322, 461, 540, 695]}, 
    {'start_frame': 868, 'end_frame': 1243, 'hit_points': [900, 959, 1011, 1129, 1150]},
    {'start_frame': 1388, 'end_frame': 1858, 'hit_points': [1400, 1448, 1510, 1600, 1700, 1780]}, 
    {'start_frame': 3275, 'end_frame': 3555, 'hit_points': [3295, 3350, 3459, 3538]}]

    # rallies = video_cutter.generate_rallies(refined_hit_points, max_frame_gap=240, buffer_frames=120)
    # print("\nGenerated rallies:")
    # for i, rally in enumerate(rallies):
    #    print(f"Rally {i + 1}: Start Frame = {rally['start_frame']}, End Frame = {rally['end_frame']}, Hit Points = {rally['hit_points']}")

    # 返回处理结果
    return {
        'output_video_frames': output_video_frames,
        'fps': fps,
        'rallies': rallies,
        'video_cutter': video_cutter,
        'video_frames': video_frames  # 添加原始视频帧，用于生成rallies
    }

if __name__ == "__main__":
    result = main()
    # 保存完整视频
    save_video(result['output_video_frames'], "output_videos/output_video.avi", fps=result['fps'])
    # 保存rallies视频
    result['video_cutter'].save_rallies_as_videos(
        "output_videos/output_video.avi",
        result['rallies'],
        "output_videos/output_rallies"
    )
    # 转码完整视频为h264 mp4
    convert_to_h264(
        "output_videos/output_video.avi",
        "output_videos/output_video_h264.mp4"
    )
    # 转码rallies视频为h264 mp4
    import glob
    rallies_src_dir = "output_videos/output_rallies"
    rallies_dst_dir = "output_videos/output_rallies_h264"
    import os
    os.makedirs(rallies_dst_dir, exist_ok=True)
    for src_path in glob.glob(os.path.join(rallies_src_dir, '*.mp4')):
        base = os.path.basename(src_path)
        dst_path = os.path.join(rallies_dst_dir, base)
        convert_to_h264(src_path, dst_path)