from ultralytics import YOLO 
import cv2
import pickle
import sys
import numpy as np
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def convert_to_court_coordinates(self, pixel_point, court_keypoints):
        """
        将像素坐标转换为球场实际坐标
        :param pixel_point: (x, y) 像素坐标
        :param court_keypoints: 球场关键点列表
        :return: (x, y) 球场实际坐标
        """
        # 获取球场边界点
        court_width = 23.77  # 标准网球场的宽度（米）
        court_length = 10.97  # 标准网球场的半场长度（米）
        
        # 获取球场四个角点（像素坐标）
        top_left = (court_keypoints[0], court_keypoints[1])
        top_right = (court_keypoints[2], court_keypoints[3])
        bottom_left = (court_keypoints[4], court_keypoints[5])
        bottom_right = (court_keypoints[6], court_keypoints[7])
        
        # 计算变换矩阵
        src_points = np.float32([top_left, top_right, bottom_left, bottom_right])
        dst_points = np.float32([
            [0, 0],  # 左上角
            [court_width, 0],  # 右上角
            [0, court_length],  # 左下角
            [court_width, court_length]  # 右下角
        ])
        
        # 计算透视变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 转换坐标
        point = np.float32([[pixel_point[0], pixel_point[1]]])
        transformed_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), transform_matrix)
        
        return transformed_point[0][0]  # 返回(x, y)坐标

    def calculate_actual_distance(self, point, line_start, line_end):
        """
        计算点到线段的实际距离（米）
        :param point: (x, y) 球场实际坐标
        :param line_start: (x, y) 线段起点
        :param line_end: (x, y) 线段终点
        :return: 实际距离（米）
        """
        # 计算点到线段的距离
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 计算线段长度
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 计算点到线段的投影点
        if line_length == 0:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
            
        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length**2)
        t = max(0, min(1, t))  # 限制在0到1之间
        
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        # 计算实际距离
        distance = np.sqrt((x - projection_x)**2 + (y - projection_y)**2)
        return distance

    def filter_moving_players(self, player_positions_history, min_movement_distance=5.0, min_players=2):
        """
        根据最近 120 帧的移动距离，筛选出移动距离大于阈值的球员 ID。
        确保返回的球员 ID 至少有 min_players 个。
        """
        moving_players = {}
        for track_id, positions in player_positions_history.items():
            if len(positions) > 1:
                # 计算最近 120 帧的移动距离
                total_distance = sum(
                    np.sqrt((positions[i][0] - positions[i - 1][0]) ** 2 + (positions[i][1] - positions[i - 1][1]) ** 2)
                    for i in range(1, len(positions))
                )
                moving_players[track_id] = total_distance

                # 打印移动距离
                print(f"  Player ID {track_id}: Total movement in last 120 frames = {total_distance:.2f} pixels")

        # 筛选出移动距离大于阈值的球员 ID
        moving_player_ids = {track_id for track_id, distance in moving_players.items() if distance > min_movement_distance}

        # 如果筛选后球员数量不足 min_players 个，则保留移动距离最大的前 min_players 个球员
        if len(moving_player_ids) < min_players:
            sorted_players = sorted(moving_players.items(), key=lambda x: x[1], reverse=True)
            moving_player_ids = {track_id for track_id, _ in sorted_players[:min_players]}

        return moving_player_ids

    def calculate_closest_players(self, court_keypoints, player_dict, moving_player_ids):
        """
        根据距离底线的距离，计算离底线 1 和底线 2 最近的球员。
        """
        # 定义底线 1 和底线 2 的起点和终点
        baseline1_start, baseline1_end = (0, 0), (23.77, 0)  # 底线 1
        baseline2_start, baseline2_end = (0, 10.97), (23.77, 10.97)  # 底线 2

        closest_to_baseline1 = {"player_id": None, "distance": float('inf')}
        closest_to_baseline2 = {"player_id": None, "distance": float('inf')}

        for track_id, bbox in player_dict.items():
            if track_id not in moving_player_ids:
                continue  # 跳过未移动的球员

            # 获取球员中心点
            player_center = get_center_of_bbox(bbox)

            # 转换为球场实际坐标
            court_position = self.convert_to_court_coordinates(player_center, court_keypoints)

            # 计算到底线 1 和底线 2 的距离
            distance_to_baseline1 = self.calculate_actual_distance(court_position, baseline1_start, baseline1_end)
            distance_to_baseline2 = self.calculate_actual_distance(court_position, baseline2_start, baseline2_end)

            # 更新离底线 1 最近的球员
            if distance_to_baseline1 < closest_to_baseline1["distance"]:
                closest_to_baseline1["player_id"] = track_id
                closest_to_baseline1["distance"] = distance_to_baseline1

            # 更新离底线 2 最近的球员
            if distance_to_baseline2 < closest_to_baseline2["distance"]:
                closest_to_baseline2["player_id"] = track_id
                closest_to_baseline2["distance"] = distance_to_baseline2

        return closest_to_baseline1, closest_to_baseline2

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        遍历 player_detections 中的球员 ID，逐帧选择距离底线 1 最近的球员作为 Player 1，
        距离底线 2 最近的球员作为 Player 2，并动态绑定球员 ID。
        首先筛选出在一定时间内（如 120 帧）内有移动的球员，然后根据距离进行二次筛选。
        """
        # 用于存储绑定的 Player 1 和 Player 2 的 ID
        player_bindings = {1: None, 2: None}

        # 存储每帧的 Player 1 和 Player 2 的检测结果
        filtered_player_detections = []

        # 存储每个球员的历史位置，用于检测移动
        player_positions_history = {}

        for frame_idx, player_dict in enumerate(player_detections):
            print(f"Frame {frame_idx}:")

            # 更新球员位置历史
            for track_id, bbox in player_dict.items():
                # 获取球员中心点
                player_center = get_center_of_bbox(bbox)

                # 如果该球员 ID 不在历史记录中，初始化
                if track_id not in player_positions_history:
                    player_positions_history[track_id] = []

                # 添加当前帧的球员位置到历史记录
                player_positions_history[track_id].append(player_center)

                # 保留最近 120 帧的历史位置
                if len(player_positions_history[track_id]) > 120:
                    player_positions_history[track_id].pop(0)

            # 调用动态检测逻辑，筛选出移动的球员
            moving_player_ids = self.filter_moving_players(player_positions_history, min_movement_distance=100, min_players=2)

            # 调用距离计算逻辑，筛选出离底线最近的球员
            closest_to_baseline1, closest_to_baseline2 = self.calculate_closest_players(
                court_keypoints, player_dict, moving_player_ids
            )

            # 动态绑定 Player 1 和 Player 2 的 ID
            if closest_to_baseline1["player_id"] is not None:
                player_bindings[1] = closest_to_baseline1["player_id"]
            if closest_to_baseline2["player_id"] is not None:
                player_bindings[2] = closest_to_baseline2["player_id"]

            # 打印当前帧的绑定信息
            print(f"  Player 1 (closest to baseline 1): ID {player_bindings[1]}, Distance = {closest_to_baseline1['distance']:.2f} m")
            print(f"  Player 2 (closest to baseline 2): ID {player_bindings[2]}, Distance = {closest_to_baseline2['distance']:.2f} m")

            # 构造当前帧的 Player 1 和 Player 2 的检测结果
            frame_detection = {}
            if player_bindings[1] in player_dict:
                frame_detection[1] = player_dict[player_bindings[1]]  # Player 1
            if player_bindings[2] in player_dict:
                frame_detection[2] = player_dict[player_bindings[2]]  # Player 2

            # 将当前帧的结果添加到输出列表
            filtered_player_detections.append(frame_detection)

        return filtered_player_detections

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
        # 检查 box.id 是否为 None
            if box.id is None or box.cls is None or box.xyxy is None:
                continue  # 跳过无效的检测框
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    # 以下两个函数暂未想好逻辑如何使用

    def extract_color_histogram(self, frame, bbox):
        """
        提取检测框内的颜色直方图。
        :param frame: 当前帧图像。
        :param bbox: 检测框 [x1, y1, x2, y2]。
        :return: 归一化的颜色直方图。
        """
        x1, y1, x2, y2 = map(int, bbox)  # 确保坐标为整数
        if x2 <= x1 or y2 <= y1:  # 检查检测框的宽度和高度是否有效
            print(f"无效的检测框: {bbox}")
            return np.zeros((50, 60), dtype=np.float32)  # 返回空直方图

        player_roi = frame[y1:y2, x1:x2]  # 提取检测框内的图像
        if player_roi.size == 0:  # 检查 ROI 是否为空
            print(f"检测框提取失败: {bbox}")
            return np.zeros((50, 60), dtype=np.float32)  # 返回空直方图

        hsv_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)  # 转换为 HSV 颜色空间
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])  # 计算颜色直方图
        cv2.normalize(hist, hist)  # 归一化直方图
        return hist

    def match_players(self, frame, player_detections):
        """
        根据颜色特征匹配球员并分配 ID。
        :param frame: 当前帧图像。
        :param player_detections: 当前帧的球员检测结果，格式为 {player_id: [x1, y1, x2, y2], ...}。
        :return: 带有分配 ID 的检测结果列表，格式为 [(player_id, bbox), ...]。
        """
        if not hasattr(self, 'players'):
            self.players = {}

        if not isinstance(player_detections, dict):
            raise ValueError(f"player_detections 应为字典格式，但接收到 {type(player_detections)}")

        if not player_detections:
            print("未检测到任何球员，跳过匹配")
            return []

        updated_players = {}
        results = []

        for player_id, bbox in player_detections.items():
            if len(bbox) != 4:  # 检查检测框是否有效
                print(f"无效的检测框: {bbox}")
                continue

            # 提取颜色直方图
            hist = self.extract_color_histogram(frame, bbox)

            # 匹配当前检测框与已有球员
            best_match_id = None
            best_match_score = float('-inf')

            for existing_id, player_hist in self.players.items():
                score = cv2.compareHist(hist, player_hist, cv2.HISTCMP_CORREL)  # 计算直方图相似性
                if score > best_match_score and score > 0.5:  # 设置相似性阈值
                    best_match_id = existing_id
                    best_match_score = score

            if best_match_id is not None:
                # 如果找到匹配的球员，更新其特征
                updated_players[best_match_id] = hist
                results.append((best_match_id, bbox))
            else:
                # 如果没有匹配到，分配新的 ID
                updated_players[player_id] = hist
                results.append((player_id, bbox))

        # 更新球员特征
        self.players = updated_players
        return results