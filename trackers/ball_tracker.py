from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np # 确保导入 numpy

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.ball_positions = []  # 存储球的位置历史
        self.max_history = 10  # 最大历史帧数
        self.min_confidence = 0.15  # 最小置信度
        self.last_ball_position = None  # 上一帧球的位置
        self.missing_frames = 0  # 连续未检测到球的帧数
        self.max_missing_frames = 5  # 最大允许的连续未检测帧数
        self.ball_speed = None  # 球的速度
        self.ball_size = None  # 球的大小
        self.search_radius = 150  # 搜索半径
        self.min_ball_size = 5  # 最小球大小
        self.max_ball_size = 50  # 最大球大小
        self.speed_threshold = 20  # 速度阈值，超过此值进行模糊处理

        # 运动模糊补偿参数
        self.speed_threshold = 25.0 # 触发模糊的速度阈值
        self.blur_kernel_size = (5, 5) # 固定模糊核

        # 多帧融合参数 (存储最近几帧的原始检测结果)
        self.detection_buffer = [] # 存储 [(bbox, conf, size), ...] 列表的列表
        self.buffer_size = 2
        self.frame_weights = [0.7, 0.3] # 最新帧权重高

        # 其他追踪参数
        self.search_radius = 150.0 # 用于关联检测结果的最大距离
        self.min_ball_size = 5   # 最小有效球大小
        self.max_ball_size = 50  # 最大有效球大小
        self.size_consistency_factor = 1.2 # 大小一致性置信度提升因子
        self.prediction_boost_factor = 1.5 # 预测位置置信度提升因子
        self.distance_penalty_factor = 0.5 # 距离过远置信度惩罚因子

    def interpolate_ball_positions(self, ball_positions):
        # ... 此方法保持不变 ...
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        # ... 此方法保持不变 ...
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        print("\n=== 开始检测球 ===")
        
        # 检查是否需要模糊处理
        if self.ball_speed:
            speed = np.sqrt(self.ball_speed[0]**2 + self.ball_speed[1]**2)
            print(f"当前速度: {speed:.2f}")
            
            if speed > self.speed_threshold:
                print("检测到高速运动，进行模糊处理")
                # 对当前帧进行轻微模糊处理
                blurred_frame = cv2.GaussianBlur(frame, (3,3), 0)
                results = self.model.predict(
                    blurred_frame,
                    conf=self.min_confidence,
                    imgsz=640,
                    verbose=False
                )[0]
            else:
                results = self.model.predict(
                    frame,
                    conf=self.min_confidence,
                    imgsz=640,
                    verbose=False
                )[0]
        else:
            results = self.model.predict(
                frame,
                conf=self.min_confidence,
                imgsz=640,
                verbose=False
            )[0]
        
        print(f"检测到 {len(results.boxes)} 个目标")
        
        ball_dict = {}
        best_conf = 0
        best_box = None
        
        # 如果有上一帧的位置，计算预测位置
        predicted_position = None
        if self.last_ball_position and self.ball_speed:
            last_center = self._get_center_of_bbox(self.last_ball_position)
            predicted_position = (
                last_center[0] + self.ball_speed[0],
                last_center[1] + self.ball_speed[1]
            )
            print(f"预测位置: {predicted_position}")
        
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            conf = box.conf.tolist()[0]
            cls_id = box.cls.tolist()[0]
            cls_name = results.names[int(cls_id)]
            
            print(f"目标: {cls_name}, 置信度: {conf:.2f}, 位置: {result}")
            
            # 检查球的大小是否合理
            x1, y1, x2, y2 = result
            width = x2 - x1
            height = y2 - y1
            size = max(width, height)
            
            if size < self.min_ball_size or size > self.max_ball_size:
                print(f"球大小不合理: {size}")
                continue
            
            # 如果是网球且置信度高于阈值
            if cls_name == "tennis ball" and conf > self.min_confidence:
                curr_center = self._get_center_of_bbox(result)
                
                # 如果是第一帧或者上一帧没有检测到球
                if self.last_ball_position is None:
                    ball_dict[1] = result
                    self.last_ball_position = result
                    self.ball_size = size
                    print(f"找到第一个网球: {result}")
                    break
                
                # 计算与上一帧球位置的距离
                if self.last_ball_position:
                    last_center = self._get_center_of_bbox(self.last_ball_position)
                    distance = np.sqrt((last_center[0] - curr_center[0])**2 + 
                                     (last_center[1] - curr_center[1])**2)
                    
                    print(f"与上一帧距离: {distance:.2f}")
                    
                    # 如果距离合理（考虑球的速度和搜索半径）
                    if distance < self.search_radius:
                        # 如果有预测位置，计算与预测位置的距离
                        if predicted_position:
                            pred_distance = np.sqrt((predicted_position[0] - curr_center[0])**2 + 
                                                  (predicted_position[1] - curr_center[1])**2)
                            print(f"与预测位置距离: {pred_distance:.2f}")
                            # 如果距离预测位置较近，增加置信度
                            if pred_distance < self.search_radius / 2:
                                conf *= 1.2
                                print(f"增加置信度: {conf:.2f}")
                        
                        # 如果球的大小变化不大，增加置信度
                        if self.ball_size and abs(size - self.ball_size) < 10:
                            conf *= 1.1
                            print(f"大小一致，增加置信度: {conf:.2f}")
                        
                        if conf > best_conf:
                            best_conf = conf
                            best_box = result
                            print(f"更新最佳检测: {result}, 置信度: {conf:.2f}")
        
        # 如果找到了合理的球位置
        if best_box is not None:
            ball_dict[1] = best_box
            curr_center = self._get_center_of_bbox(best_box)
            last_center = self._get_center_of_bbox(self.last_ball_position)
            
            # 更新球的速度
            self.ball_speed = (
                curr_center[0] - last_center[0],
                curr_center[1] - last_center[1]
            )
            
            self.last_ball_position = best_box
            self.ball_size = max(best_box[2] - best_box[0], best_box[3] - best_box[1])
            self.missing_frames = 0
            print(f"更新球位置: {best_box}, 速度: {self.ball_speed}")
        else:
            # 如果连续几帧没有检测到球，尝试使用预测位置
            if self.last_ball_position and self.missing_frames < self.max_missing_frames:
                if predicted_position:
                    # 使用预测位置创建边界框
                    x, y = predicted_position
                    size = self.ball_size if self.ball_size else 20
                    predicted_box = [
                        x - size/2, y - size/2,
                        x + size/2, y + size/2
                    ]
                    ball_dict[1] = predicted_box
                    print(f"使用预测位置: {predicted_box}")
                else:
                    ball_dict[1] = self.last_ball_position
                    print(f"使用上一帧球位置: {self.last_ball_position}")
                self.missing_frames += 1
                print(f"连续未检测帧数: {self.missing_frames}")
            else:
                self.last_ball_position = None
                self.ball_speed = None
                self.ball_size = None
                self.missing_frames = 0
                print("未检测到球，重置状态")
        
        print("=== 检测结束 ===\n")
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        # ... 此方法保持不变 ...
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # 检查 bbox 是否有效 (例如，非 None 或空列表)
                if bbox and len(bbox) == 4:
                    try:
                        cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                    except Exception as e:
                        print(f"Error drawing bbox: {bbox}, error: {e}") # 添加错误处理
            output_video_frames.append(frame)

        return output_video_frames

    # 添加辅助方法
    def _get_center_of_bbox(self, bbox):
        """获取边界框的中心点"""
        if not bbox or len(bbox) != 4:
            return (0, 0) # 返回默认值或抛出错误
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)