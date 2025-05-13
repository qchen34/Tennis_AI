import numpy as np

import cv2
import os

class VideoCut:
    def __init__(self, ball_detections, ball_shot_frames, acceleration_threshold=5.0, frame_merge_threshold=60):
        """
        初始化 VideoCut 类的实例。
        :param ball_detections: 包含每帧网球检测结果的列表，例如 [{1: [x1, y1, x2, y2]}, ...]。
        :param ball_shot_frames: 初步判断的击球点帧数列表。
        :param acceleration_threshold: 网球加速度变化的阈值，用于二次筛选击球点。
        :param frame_merge_threshold: 帧数合并的阈值，距离较近的击球点取平均值。
        """
        self.ball_positions = self._convert_detections_to_positions(ball_detections)
        self.ball_shot_frames = ball_shot_frames
        self.acceleration_threshold = acceleration_threshold
        self.frame_merge_threshold = frame_merge_threshold

    def _convert_detections_to_positions(self, ball_detections):
        """
        将 ball_detections 转换为 ball_positions，提取网球的中心点 (cx, cy)。
        :param ball_detections: 包含每帧网球检测结果的列表。
        :return: 包含每帧网球中心点的列表，例如 [(cx1, cy1), (cx2, cy2), ...]。
        """
        ball_positions = []
        for detection in ball_detections:
            if 1 in detection:  # 检查是否检测到网球
                x1, y1, x2, y2 = detection[1]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                ball_positions.append((cx, cy))
            else:
                ball_positions.append(None)  # 如果未检测到网球，填充 None
        return ball_positions

    def refine_hit_points(self):
        """
        根据网球加速度和速度变化对初步击球点进行二次筛选，并合并相近的击球点。
        :return: 优化后的击球点帧数列表。
        """
        refined_hit_points = []

        # 使用滑动窗口计算速度
        window_size = 5
        velocities = []
        for i in range(window_size, len(self.ball_positions)):
            if any(pos is None for pos in self.ball_positions[i-window_size:i]):
                velocities.append(0)
                continue

            # 计算窗口内的平均速度
            speed_sum = 0
            for j in range(i-window_size+1, i):
                x1, y1 = self.ball_positions[j-1]
                x2, y2 = self.ball_positions[j]
                speed_sum += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            velocities.append(speed_sum / (window_size-1))

        # 动态调整加速度阈值
        for frame in self.ball_shot_frames:
            if frame > window_size and frame < len(velocities):
                # 计算动态阈值
                dynamic_threshold = np.mean(velocities[max(0, frame-10):frame]) * 0.5
                
                # 计算加速度
                acceleration = velocities[frame] - velocities[frame-1]

                # 判断是否满足击球条件
                if abs(acceleration) > dynamic_threshold:
                    refined_hit_points.append(frame)
                else:
                    # 检查速度变化趋势
                    if frame > window_size+1:
                        speed_change = velocities[frame] - velocities[frame-2]
                        if abs(speed_change) > dynamic_threshold * 0.7:
                            refined_hit_points.append(frame)

        # 改进的合并逻辑
        refined_hit_points = sorted(refined_hit_points)
        merged_hit_points = []
        temp_group = [refined_hit_points[0]]

        for i in range(1, len(refined_hit_points)):
            if refined_hit_points[i] - temp_group[-1] <= self.frame_merge_threshold:
                temp_group.append(refined_hit_points[i])
            else:
                # 使用加权平均，权重为速度大小
                weights = [velocities[f] for f in temp_group]
                merged_hit_points.append(int(np.average(temp_group, weights=weights)))
                temp_group = [refined_hit_points[i]]

        if temp_group:
            weights = [velocities[f] for f in temp_group]
            merged_hit_points.append(int(np.average(temp_group, weights=weights)))

        return merged_hit_points

    def generate_rallies(self, refined_hit_points, max_frame_gap=240, buffer_frames=120):
        """
        根据优化后的击球点生成 rallies（回合）。
        :param refined_hit_points: 优化后的击球点帧数列表。
        :param max_frame_gap: 最大帧间隔（超过此值则判定为新 rally）。
        :param buffer_frames: 每个 rally 的起始和结束缓冲帧数。
        :return: 包含每个 rally 的列表，每个 rally 包含 start_frame, end_frame 和 hit_points。
        """
        rallies = []
        if not refined_hit_points:
            return rallies

        # 初始化第一个 rally
        current_rally = {
            "start_frame": max(0, refined_hit_points[0] - buffer_frames),  # 起始帧，预留缓冲
            "end_frame": refined_hit_points[0] + buffer_frames,           # 结束帧，预留缓冲
            "hit_points": [refined_hit_points[0]]
        }

        # 遍历剩余的击球点
        for i in range(1, len(refined_hit_points)):
            hit_point = refined_hit_points[i]

            # 如果当前击球点与上一个击球点的间隔超过 max_frame_gap，则判定为新 rally
            if hit_point - current_rally["end_frame"] > max_frame_gap:
                rallies.append(current_rally)  # 保存当前 rally
                current_rally = {
                    "start_frame": max(0, hit_point - buffer_frames),  # 起始帧，预留缓冲
                    "end_frame": hit_point + buffer_frames,           # 结束帧，预留缓冲
                    "hit_points": [hit_point]
                }
            else:
                # 合并到当前 rally
                current_rally["end_frame"] = max(current_rally["end_frame"], hit_point + buffer_frames)
                current_rally["hit_points"].append(hit_point)

        # 保存最后一个 rally
        rallies.append(current_rally)

        return rallies
    
    def save_rallies_as_videos(self, video_path, rallies, output_folder):
        """
        将区分好的 rallies 保存为不同的视频文件。
        :param video_path: 输入视频文件的路径。
        :param rallies: 包含每个 rally 的列表，每个 rally 包含 start_frame, end_frame 和 hit_points。
        :param output_folder: 输出视频文件的保存路径。
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for i, rally in enumerate(rallies):
            start_frame = rally["start_frame"]
            end_frame = rally["end_frame"]

            # 设置输出视频文件路径
            output_path = os.path.join(output_folder, f"rally_{i + 1}.mp4")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            # 跳转到开始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # 逐帧读取并写入输出视频
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()  # 释放当前视频写入器

        cap.release()  # 释放视频捕获器