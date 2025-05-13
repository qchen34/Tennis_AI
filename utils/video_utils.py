import cv2

def read_video(video_path):
    """
    读取视频文件并返回帧列表
    :param video_path: 视频文件路径
    :return: 帧列表
    """
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的原始帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频原始帧率: {fps} FPS")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"读取完成，总帧数: {len(frames)}")
    return frames, fps

def save_video(output_video_frames, output_video_path, fps=None):
    """
    保存视频帧为视频文件
    :param output_video_frames: 要保存的视频帧列表
    :param output_video_path: 输出视频的路径
    :param fps: 视频帧率，必须为正数
    :return: None
    """
    if fps is None or fps <= 0:
        raise ValueError(f"保存视频时fps无效（fps={fps}），请确保传入正确的帧率！")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        fps, 
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"视频保存完成，总帧数: {len(output_video_frames)}，帧率: {fps} FPS")