import pandas as pd
import numpy as np
from copy import deepcopy
import constants
from utils import measure_distance, convert_pixel_distance_to_meters

def compute_player_stats(ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, mini_court, video_frames_length, fps=24):
    """
    计算球员的统计数据
    :param ball_shot_frames: 击球帧列表
    :param ball_mini_court_detections: 球在迷你球场上的坐标
    :param player_mini_court_detections: 球员在迷你球场上的坐标
    :param mini_court: 迷你球场对象
    :param video_frames_length: 视频总帧数
    :return: 包含球员统计数据的DataFrame
    """
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    }]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]

                # 检查数据有效性
        if (start_frame >= len(ball_mini_court_detections) or 
            end_frame >= len(ball_mini_court_detections) or
            not ball_mini_court_detections[start_frame] or 
            not ball_mini_court_detections[end_frame] or
            1 not in ball_mini_court_detections[start_frame] or
            1 not in ball_mini_court_detections[end_frame]):
            print(f"Warning: 帧 {start_frame} 到 {end_frame} 的球位置数据无效，跳过此击球")
            continue



        ball_shot_time_in_seconds = (end_frame-start_frame)/fps # 24fps

        # Get distance covered by the ball
        try:
            distance_covered_by_ball_pixels = measure_distance(
                ball_mini_court_detections[start_frame][1],
                ball_mini_court_detections[end_frame][1]
            )
        except (KeyError, IndexError) as e:
            print(f"Error: 计算球距时出错，帧 {start_frame}-{end_frame}: {e}")
            continue
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who hit the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(), 
            key=lambda player_id: measure_distance(
                player_positions[player_id],
                ball_mini_court_detections[start_frame][1]
            )
        )

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id]
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    # 创建DataFrame并处理数据
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(video_frames_length))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # 计算平均值
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']

    return player_stats_data_df