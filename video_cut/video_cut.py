import cv2
import os

class VideoCut:
    def __init__(self, video_path, output_folder, ball_shot_frames, pre_frames=30, post_frames=30, frame_gap=100):
        """
        初始化 VideoCut 类的实例。
        :param video_path: 输入视频文件的路径。
        :param output_folder: 保存输出视频剪辑的文件夹。
        :param ball_shot_frames: 包含击球点帧数的列表，例如 [100, 300, 500]。
        :param pre_frames: 每个击球片段起点距离击球点的帧数（默认 30 帧）。
        :param post_frames: 每个击球片段终点距离击球点的帧数（默认 30 帧）。
        :param frame_gap: 判断击球点间隔的帧数阈值（默认 100 帧）。
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.ball_shot_frames = ball_shot_frames
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.frame_gap = frame_gap
        self.hit_points = self.calculate_cut_points()
        # 根据击球点帧数计算剪辑点列表。

    def calculate_cut_points(self):
        """
        根据击球点帧数计算剪辑点列表。
        :return: 包含 (start_frame, end_frame) 的元组列表。
        """
        cut_points = []
        for i, frame in enumerate(self.ball_shot_frames):
            # 计算起点帧
            start_frame = max(0, frame - self.pre_frames)

            # 判断终点帧
            if i < len(self.ball_shot_frames) - 1:
                next_frame = self.ball_shot_frames[i + 1]
                if next_frame - frame > self.frame_gap:
                    # 如果与下一个击球点间隔超过 frame_gap，则终点为当前击球点后的 post_frames。
                    end_frame = min(next_frame - 1, frame + self.post_frames)
                else:
                    # 如果间隔小于等于 frame_gap，则终点为当前击球点与下一个击球点的中间帧。
                    end_frame = (frame + next_frame) // 2
            else:
                # 如果是最后一个击球点，则终点为当前击球点后的 post_frames。
                end_frame = frame + self.post_frames

            cut_points.append((start_frame, end_frame))

        return cut_points

    def cut_video(self):
        # 定义方法 cut_video，用于根据击球时间段剪辑视频。

        if not os.path.exists(self.output_folder):
            # 如果输出文件夹不存在，则创建它。
            os.makedirs(self.output_folder)

        cap = cv2.VideoCapture(self.video_path)
        # 打开输入视频文件。

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频的帧率（每秒帧数）。

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 获取视频的总帧数。

        duration = frame_count / fps
        # 计算视频的总时长（秒）。

        print(f"Video Duration: {duration}s, FPS: {fps}, Total Frames: {frame_count}")
        # 打印视频的时长、帧率和总帧数。

        for idx, (start, end) in enumerate(self.hit_points):
            # 遍历击球时间段列表，每个时间段包含起始时间和结束时间。

            start_frame = int(start * fps)
            # 将起始时间转换为帧索引。

            end_frame = int(end * fps)
            # 将结束时间转换为帧索引。

            output_path = os.path.join(self.output_folder, f"shot{idx + 1}.mp4")
            # 生成输出视频剪辑的文件路径，命名为 shot1.mp4, shot2.mp4 等。

            self._write_clip(cap, start_frame, end_frame, output_path)
            # 调用私有方法 _write_clip，将指定帧范围的视频写入输出文件。

        cap.release()
        # 释放视频捕获对象。

    def _write_clip(self, cap, start_frame, end_frame, output_path):
        # 定义私有方法 _write_clip，用于将指定帧范围的视频写入输出文件。

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # 设置视频捕获对象的当前帧位置为起始帧。

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 定义视频编码格式为 mp4v。

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # 获取视频的帧率。

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取视频帧的宽度。

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取视频帧的高度。

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        # 创建一个 VideoWriter 对象，用于写入视频文件。

        for frame_idx in range(start_frame, end_frame):
            # 遍历指定帧范围内的每一帧。

            ret, frame = cap.read()
            # 读取当前帧。

            if not ret:
                # 如果读取失败，则退出循环。
                break

            out.write(frame)
            # 将当前帧写入输出视频文件。

        out.release()
        # 释放 VideoWriter 对象。