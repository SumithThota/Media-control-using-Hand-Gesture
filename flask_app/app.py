from video_feed import generate_video
from flask import Flask, render_template, request, Response
import json

app = Flask(__name__)

STATE_PATH = 'C:/Users/thota/Desktop/Minor Project/PRO/gesture_based_youtube_control-master/data/player_state.json'

def get_video_id(ytb_link):
    """
    Extract a video ID from the provided link.
    """
    idx_pattern = ytb_link.find('?v=')
    if idx_pattern == -1:
        return None  # Return None if the pattern is not found
    ytb_id = ytb_link[idx_pattern + 3: idx_pattern + 3 + 11]
    return ytb_id

# Demo page
@app.route('/', methods=['POST', 'GET'])
def demo():
    if request.method == 'POST':
        ytb_link = request.form.get('link')
        video_id = get_video_id(ytb_link)
        if video_id:
            return render_template('demo.html', video_id=video_id)
        else:
            return render_template('demo.html', error="Invalid YouTube link.")

    return render_template('demo.html')

# Access YouTube video info
@app.route('/video_info', methods=['POST'])
def get_video_info():
    output = request.get_json()
    with open(STATE_PATH, 'w') as outfile:
        json.dump(output, outfile)

    return '', 204  # Return no content

# Serve webcam video
@app.route('/webcam')
def stream_video():
    return Response(generate_video(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

