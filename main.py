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


def main():
    print("开始执行main函数...")
    
    # Read Video
    print("1. 开始读取视频...")
    input_video_path = "input/tennis_match_hardcourt_short.mp4"
    video_frames, fps = read_video(input_video_path)
    print(f"视频读取完成，共{len(video_frames)}帧")

    # Detect Players and Ball
    print("2. 初始化球员跟踪器...")
    player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
    print("3. 初始化球跟踪器...")
    ball_tracker = BallTracker(model_path='models/yolo9_best.pt')
    
    print("4. 开始检测球员...")
    player_detections = player_tracker.detect_frames(video_frames, 
                                                    read_from_stub=False, 
                                                    stub_path="tracker_stubs/player_detections.pkl")
    print(f"球员检测完成，检测到{len(player_detections)}帧的球员")

    print("5. 开始检测球...")
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                read_from_stub=False, 
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

    print("9. 开始选择和过滤球员...")
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    print(f"球员选择和过滤完成，剩余{len(player_detections)}帧的球员数据")

    # Mini Court
    print("10. 初始化迷你球场...")
    mini_court = MiniCourt(video_frames[0]) 

    # Detect ball shots
    print("11. 开始检测击球帧...")
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(f"击球帧检测完成: {ball_shot_frames}")

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints)

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
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
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
    save_video(output_video_frames, "output_videos/output_video.avi")
    print("14. 所有处理完成！")

if __name__ == "__main__":
    main()