from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
from config import S3_BUCKET, S3_LOCAL_DIR

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://chenqiwei.org"}})

@app.route('/process_highlight', methods=['POST'])
def process_highlight():
    data = request.json
    userid = data.get('userid')
    timeslot = data.get('timeslot')
    video_id_list = data.get('video_id_list', [])
    
    if not userid or not timeslot or not video_id_list:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # 调用highlight_main.py处理视频
    cmd = [
        "python3", "highlight_main.py",
        "--userid", userid,
        "--timeslot", timeslot,
        "--video_ids", ",".join(video_id_list)
    ]
    try:
        subprocess.run(cmd, check=True)
        return jsonify({"status": "success"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Process failed: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 