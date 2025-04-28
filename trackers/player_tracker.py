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

    def choose_players(self, court_keypoints, player_dict):
        """
        选择距离球场底线最近的两个球员
        :param court_keypoints: 球场关键点
        :param player_dict: 球员检测结果字典
        :return: 选择的球员ID列表
        """
        if not isinstance(player_dict, dict) or not player_dict:
            return []
            
        if not isinstance(court_keypoints, (list, np.ndarray)) or len(court_keypoints) < 8:
            return []

        try:
            # 定义两条底线（球场实际坐标）
            baseline1_start = (0, 0)  # 左上角
            baseline1_end = (23.77, 0)  # 右上角
            baseline2_start = (0, 10.97)  # 左下角
            baseline2_end = (23.77, 10.97)  # 右下角
            
            player_distances = []
            
            for track_id, bbox in player_dict.items():
                # 获取球员中心点
                player_center = get_center_of_bbox(bbox)
                
                # 转换为球场实际坐标
                court_position = self.convert_to_court_coordinates(player_center, court_keypoints)
                
                # 计算到两条底线的实际距离
                distance_to_baseline1 = self.calculate_actual_distance(
                    court_position, baseline1_start, baseline1_end
                )
                distance_to_baseline2 = self.calculate_actual_distance(
                    court_position, baseline2_start, baseline2_end
                )
                
                # 取较小值作为该球员的距离
                min_distance = min(distance_to_baseline1, distance_to_baseline2)
                player_distances.append((track_id, min_distance))
            
            if not player_distances:
                return []
                
            # 选择距离最小的两个球员
            player_distances.sort(key=lambda x: x[1])
            chosen_players = [player_distances[0][0], player_distances[1][0]]
            return chosen_players
            
        except Exception as e:
            print(f"选择球员时发生错误: {str(e)}")
            return []

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
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

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
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


    