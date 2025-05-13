import cv2
import json
import os
from config import S3_LOCAL_DIR

def generate_jpg_from_video(video_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(S3_LOCAL_DIR, os.path.basename(video_path).replace('.mp4', '.jpg'))
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()
    return output_path

def generate_json_for_rallies(rallies_dir, output_json_path, base_url, video_id_prefix="r_"):
    rallies_data = {}
    for filename in os.listdir(rallies_dir):
        if filename.endswith('.mp4'):
            video_id = f"{video_id_prefix}{filename.replace('.mp4', '')}"
            video_url = f"{base_url}/{filename}"
            rallies_data[video_id] = {
                "video_url": video_url,
                "s3_key": f"rallies/{filename}"
            }
    with open(output_json_path, 'w') as f:
        json.dump(rallies_data, f, indent=2)
    return output_json_path