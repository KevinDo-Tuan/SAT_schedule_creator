from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template, session
import os
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import google.generativeai as genai
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import base64
from io import BytesIO
import re
import socket
import uuid
import time
from functools import wraps
import pandas as pd

# Import template filters
from .filters.template_filters import register_template_filters

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='../frontend', static_url_path='', template_folder='templates')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recorded_scores')
SCHEDULES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_schedules')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SCHEDULES_FOLDER, exist_ok=True)

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure session
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-123')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Store processing status (in production, use Redis or a database)
processing_status = {}

def get_schedule_path(schedule_id):
    """Get the path to a schedule file"""
    return os.path.join(SCHEDULES_FOLDER, f"{schedule_id}.json")

def save_schedule(schedule_data):
    """Save schedule data to a file and return the schedule ID"""
    schedule_id = str(uuid.uuid4())
    schedule_path = get_schedule_path(schedule_id)
    
    with open(schedule_path, 'w', encoding='utf-8') as f:
        json.dump(schedule_data, f, ensure_ascii=False, indent=2)
    
    return schedule_id

def get_schedule(schedule_id):
    """Retrieve a schedule by ID"""
    schedule_path = get_schedule_path(schedule_id)
    if not os.path.exists(schedule_path):
        return None
    
    with open(schedule_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_latest_pdf():
    """Get the most recently created PDF file from the recorded_scores directory"""
    try:
        pdf_files = glob.glob(os.path.join(RECORDED_SCORES_FOLDER, '*.pdf'))
        if not pdf_files:
            return None
        # Sort by creation time (newest first) and return the first one
        latest_pdf = max(pdf_files, key=os.path.getctime)
        return latest_pdf
    except Exception as e:
        print(f"Error finding latest PDF: {str(e)}")
        return None

def generate_sat_schedule(score_data):
    """Generate a structured SAT study schedule based on score data"""
    try:
        # Initialize Gemini model
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        
        # Read the prompt template
        with open('prompt.md', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Format the prompt with the score data
        prompt = f"""{prompt_template}
        
        Here is the student's score data:
        {score_data}
        
        Please generate a detailed SAT study schedule based on this information.
        """
        
        # Generate the schedule
        response = model.generate_content(prompt)
        
        # Parse the response into a structured format
        schedule = {
            "sections": [],
            "practice_tests": [],
            "resources": [],
            "timeline": []
        }
        
        # Basic parsing of the response (this can be enhanced based on your needs)
        text = response.text
        
        # Extract sections (this is a simplified example)
        if "Reading and Writing" in text:
            schedule["sections"].append({
                "name": "Reading and Writing",
                "focus_areas": [],
                "study_materials": []
            })
        
        if "Math" in text:
            schedule["sections"].append({
                "name": "Math",
                "focus_areas": [],
                "study_materials": []
            })
        
        # Add more parsing logic here based on your specific needs
        
        return schedule
        
    except Exception as e:
        print(f"Error generating SAT schedule: {str(e)}")
        return None

def process_image_to_text(image_path):
    """Extract text from image using OCR or Gemini for PDFs"""
    try:
        if not os.path.exists(image_path):
            return {"error": "File not found"}
            
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Process images with Tesseract
            try:
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img, lang='eng')
                if not text.strip():
                    return {"error": "No text could be extracted from the image"}
                return {"text": text, "source": "tesseract"}
                
            except Exception as e:
                return {"error": f"Error processing image with Tesseract: {str(e)}"}
                
        elif image_path.lower().endswith('.pdf'):
            # Process PDFs with Gemini Vision
            try:
                with open(image_path, 'rb') as pdf_file:
                    pdf_data = pdf_file.read()
                model = genai.GenerativeModel('gemini-pro-vision')
                response = model.generate_content(["Extract all text from this document:", pdf_data])
                if not response.text.strip():
                    return {"error": "No text could be extracted from the PDF"}
                return {"text": response.text, "source": "gemini-vision"}
                
            except Exception as e:
                return {"error": f"Error processing PDF with Gemini: {str(e)}"}
        else:
            return {"error": "Unsupported file format. Please upload a PNG, JPG, or PDF file."}
            
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Configure Gemini
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)
path_to_prompt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.md')

# Register template filters
register_template_filters(app)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

def generate_schedule(text):
    """Generate SAT study schedule using Gemini"""
    try:
        # Load the prompt from file
        with open(path_to_prompt, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Send the prompt and text to Gemini
        response = model.generate_content([prompt_text, text])
        
        # Extract JSON from the response
        json_str = response.text
        
        # Clean the JSON string (remove markdown code blocks if present)
        json_str = re.sub(r'```json\n|```', '', json_str).strip()
        
        # Parse the JSON response
        schedule = json.loads(json_str)
        
        # Add timestamp
        schedule['generated_at'] = datetime.now().isoformat()
        
        return {"schedule": schedule, "status": "success"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse schedule: {str(e)}", "raw_response": json_str}
    except Exception as e:
        return {"error": f"Error generating schedule: {str(e)}"}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Save uploaded file and return its path"""
    if not file or file.filename == '':
        return None, "No file selected"
        
    if not allowed_file(file.filename):
        return None, "File type not allowed"
    
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"upload_{timestamp}.{file_ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        # Save the file
        file.save(filepath)
        return filepath, None
    except Exception as e:
        return None, f"Error saving file: {str(e)}"

@app.route('/')
def index():
    """Serve the main application page"""
    return app.send_static_file('upload.html')

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload only a PNG, JPG, or PDF file.'}), 400
    
    try:
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Save the uploaded file with a unique name
        filename = secure_filename(f"{request_id}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get the latest PDF (which should be the one just uploaded)
        latest_pdf = get_latest_pdf()
        if not latest_pdf:
            return jsonify({'error': 'Failed to process the uploaded file'}), 500
        
        # Store processing status
        processing_status[request_id] = {
            'status': 'processing',
            'start_time': time.time(),
            'file_path': latest_pdf,
            'schedule_id': None,
            'error': None
        }
        
        # Start background processing
        def process_in_background():
            try:
                # Process the file
                result = process_image_to_text(latest_pdf)
                
                if 'error' in result:
                    raise Exception(result['error'])
                
                # Generate SAT schedule
                schedule = generate_sat_schedule(result.get('text', ''))
                
                if not schedule:
                    raise Exception('Failed to generate schedule')
                
                # Save the schedule
                schedule_id = save_schedule(schedule)
                
                # Update processing status
                processing_status[request_id].update({
                    'status': 'completed',
                    'schedule_id': schedule_id,
                    'end_time': time.time()
                })
                
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                processing_status[request_id].update({
                    'status': 'error',
                    'error': str(e),
                    'end_time': time.time()
                })
        
        # Start the background thread
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        # Return the request ID
        return jsonify({
            'request_id': request_id,
            'status': 'processing',
            'message': 'File uploaded and processing started',
            'check_status_url': f'/api/status/{request_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<request_id>', methods=['GET'])
def check_status(request_id):
    """Check the status of a processing request"""
    if request_id not in processing_status:
        return jsonify({'error': 'Invalid request ID'}), 404
    
    status = processing_status[request_id].copy()
    
    # Calculate processing time
    status['processing_time'] = (status.get('end_time', time.time()) - status['start_time']) * 1000  # in ms
    
    # Remove internal fields
    for field in ['start_time', 'end_time']:
        status.pop(field, None)
    
    return jsonify(status)

@app.route('/api/schedule/<schedule_id>', methods=['GET'])
def get_schedule_api(schedule_id):
    """Get a generated schedule"""
    schedule = get_schedule(schedule_id)
    if not schedule:
        return jsonify({'error': 'Schedule not found'}), 404
    
    return jsonify(schedule)

@app.route('/schedule/<schedule_id>')
def view_schedule(schedule_id):
    """Display the generated schedule"""
    schedule = get_schedule(schedule_id)
    if not schedule:
        return render_template('error.html', error='Schedule not found'), 404
    
    # Add some default values if they don't exist
    schedule.setdefault('generated_at', datetime.now().isoformat())
    schedule.setdefault('practice_test_schedule', [])
    schedule.setdefault('recommended_resources', [])
    
    return render_template('schedule.html', schedule=schedule)

# Legacy route for backward compatibility
@app.route('/upload', methods=['POST'])
def legacy_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    filepath, error = save_uploaded_file(file)
    if error:
        return jsonify({"error": error}), 400
    
    # Process the file
    ocr_result = process_image_to_text(filepath)
    if "error" in ocr_result:
        return jsonify(ocr_result), 500
    
    # Redirect to the new frontend with the extracted text
    return jsonify({
        "status": "success",
        "text": ocr_result.get("text", ""),
        "source": ocr_result.get("source", "unknown")
    })

@app.route('/waiting')
def serve_waiting():
    request_id = request.args.get('request_id')
    if not request_id or request_id not in processing_status:
        return redirect(url_for('index'))
    
    return app.send_static_file('waitingpage.html')

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
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ['REMOTE_ADDR'])
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
            result = process_image_to_text(filepath)
            
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

@app.route('/api/latest_score_table')
def get_latest_score_table():
    """Get the latest score table from recorded_scores directory"""
    try:
        # Get the most recent PDF file
        latest_pdf = get_latest_pdf()
        if not latest_pdf:
            return jsonify({'error': 'No score files found'}), 404
            
        # Extract text from the PDF
        text = process_image_to_text(latest_pdf)
        
        # Use Gemini to convert text to table format
        prompt = """Convert the following text into a clean, well-formatted HTML table. 
        If the text contains score data, organize it with appropriate headers. 
        Make sure the table is responsive and well-structured.
        \n\n""" + text
        
        response = model.generate_content(prompt)
        html_table = response.text
        
        # Clean up the response to ensure it's valid HTML
        html_table = html_table.replace('```html', '').replace('```', '').strip()
        
        return jsonify({
            'success': True,
            'html_table': html_table,
            'filename': os.path.basename(latest_pdf),
            'last_modified': time.ctime(os.path.getmtime(latest_pdf))
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/answer')
def answer():
    """Serve the answer page"""
    return send_from_directory('../frontend', 'answer.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
