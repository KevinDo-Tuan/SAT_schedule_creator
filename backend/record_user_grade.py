import os
from datetime import datetime
from flask import Flask, request

app = Flask(__name__)
UPLOAD_FOLDER = 'saved_user_grades'
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
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        return f'Saved to {path}', 200
    else:
        return 'Invalid file type', 400
