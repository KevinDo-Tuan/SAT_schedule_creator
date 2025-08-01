from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import google.generativeai as genai
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recorded_scores')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', "pdf" }

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
image_path = os.path.join(UPLOAD_FOLDER, '127_0_0_1.pdf')

# Configure Gemini
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

def get_prompt_response(text):
    """Process text with Gemini model"""
    try:
        prompt = {
            "role": "user",
            "parts": ["I have text, list for me their overall SAT score, Math score, Reading and writing. Always return a dictionary with the keys 'SAT', 'Math', 'Reading', and 'Writing'. If the information is not available, return None for that key. Here's the text: " + text]
        }
        chat = model.start_chat(history=[prompt])
        response = chat.send_message("Please provide the scores in the specified format.")
        return response.text
    except Exception as e:
        return str(e)

def process_image(image_path):
    """Process image with OCR and get scores using Gemini"""
    try:
        # Open and process image with OCR
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        
        if not text.strip():
            return {"error": "No text could be extracted from the image"}
            
        # Get response from Gemini
        response = get_prompt_response(text)
        
        # Try to parse the response as JSON
        try:
            scores = json.loads(response)
            return {"scores": scores, "raw_text": text}
        except json.JSONDecodeError:
            return {"scores": {"error": "Could not parse model response"}, "raw_text": text}
            
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_frontend():
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'frontend'), 'upload.html')

@app.route('/waiting')
def serve_waiting():
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'frontend'), 'waiting.html')

def get_client_ip():
    """Get the client's IP address"""
    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        return request.environ['REMOTE_ADDR']
    else:
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0]

@app.route('/upload-score', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'scoreImage' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['scoreImage']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get file extension in lowercase
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file and file_ext in ALLOWED_EXTENSIONS:
        try:
            # Get client IP address for filename
            client_ip = get_client_ip()
            # Clean IP address to be filesystem-safe
            safe_ip = "".join(c if c.isalnum() else "_" for c in client_ip)
            
            # Ensure the upload directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Always save as PDF
            new_filename = f"{safe_ip}.pdf"
            filepath = os.path.join(UPLOAD_FOLDER, new_filename)
            
            # If the uploaded file is a PDF, save it directly
            if file_ext == 'pdf':
                file.save(filepath)
            # Otherwise, convert it to PDF
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # For images, open with PIL and save as PDF
                img = Image.open(file)
                # Convert to RGB if necessary (for PNGs with transparency)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                    background.paste(img, img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save as PDF
                img.save(filepath, 'PDF', resolution=100.0)
            else:
                # For PDFs, just save as is
                file.save(filepath)
            
            # Process the image with OCR and Gemini
            result = process_image(filepath)
            
            # If there was an error processing the image
            if 'error' in result:
                return jsonify({
                    'message': 'File uploaded but could not be processed',
                    'error': result['error'],
                    'filename': new_filename
                }), 200
                
            # If we got scores back
            return jsonify({
                'message': 'File successfully processed',
                'filename': new_filename,
                'scores': result.get('scores', {}),
                'raw_text': result.get('raw_text', '')
            }), 200
            
        except Exception as e:
            return jsonify({
                'error': f'Error processing file: {str(e)}',
                'details': str(e)
            }), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
