from flask import Flask, request
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'recorded_scores'  # this folder should be in your backend dir
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/record', methods=['POST'])
def record_user_grade():
    if 'scoreImage' not in request.files:
        return 'No file part', 400

    file = request.files['scoreImage']
    if file.filename == '':
        return 'No selected file', 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"user_score_{timestamp}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        return f"File saved to {save_path}", 200
    else:
        return 'Invalid file type', 400

if __name__ == '__main__':
    app.run(debug=True)
