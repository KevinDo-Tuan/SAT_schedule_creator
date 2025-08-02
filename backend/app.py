from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template, session
import os
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import base64
import requests
from openai import OpenAI
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import base64
from io import BytesIO
import re
import uuid
import time
import threading

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('recorded_scores')
SCHEDULES_FOLDER = os.path.join('schedules')
ALLOWED_EXTENSIONS = {'png'}  # Only allow PNG files
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SCHEDULES_FOLDER, exist_ok=True)


# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# In-memory storage for processing status
processing_status = {}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_file(file_path):
    """Create a file using OpenAI's Files API"""
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id

def process_image_with_openai(file_path):
    """Process a PNG image using OpenAI's Vision API with file upload"""
    try:
        # Read the prompt from prompt.md using the correct path
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.md')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
            
        # Verify the file is a PNG
        if not file_path.lower().endswith('.png'):
            return {
                'success': False,
                'error': 'I am sorry but we just support PNG file, please convert to PNG file before proceed. Thank you.'
            }
        
        # Upload the file to OpenAI
        try:
            file_id = create_file(file_path)
        except Exception as e:
            return {'success': False, 'error': f'Failed to upload file, please try again, thank you: {str(e)}'}
        
        try:
            # Call the OpenAI API with the file ID
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {
                            "type": "input_image",
                            "file_id": file_id,
                        },
                    ],
                }],
            )
            
            # Extract the response text
            result_text = response.output_text
            
            try:
                # Try to parse as JSON
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    schedule_data = json.loads(result_text[json_start:json_end])
                    return {'success': True, 'schedule': schedule_data}
                return {'success': False, 'error': 'Invalid response format'}
                
            except json.JSONDecodeError as e:
                return {'success': False, 'error': f'Failed to parse response: {str(e)}'}
            
        finally:
            # Clean up - delete the uploaded file
            try:
                client.files.delete(file_id)
            except Exception as e:
                print(f"Warning: Failed to delete file {file_id}: {str(e)}, must be some problems with folder")
                
    except Exception as e:
        return {'success': False, 'error': str(e)}
@app.route('/')
def index():
    """Serve the upload page"""
    return send_from_directory('../frontend', 'upload.html')

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    """Handle file upload and processing"""
    print("Uploading new file...")
    
    if 'file' not in request.files:
        return jsonify({'error': 'there are no files'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'I am sorry, but we just receive PNG file, please convert to PNG file before proceed, thank you.'}), 400
    
    try:
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        print(f"Processing request ID: {request_id}")
        
        # Save the uploaded file with a unique name
        filename = secure_filename(f"{request_id}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        

        def process_in_background():
            #Process the uploaded file in the background
            try:
                print(f"Processing file: {filepath}")
                
                # Process the file with OpenAI
                result = process_image_with_openai(filepath)
                
                if not result['success']:
                    raise Exception(result.get('error', 'Failed to process file'))
                
                # Save the schedule
                schedule_id = str(uuid.uuid4())
                schedule_path = os.path.join(SCHEDULES_FOLDER, f"{schedule_id}.json")
                with open(schedule_path, 'w') as f:
                    json.dump(result['schedule'], f, indent=2)
                
                # Update processing status
                processing_status[request_id].update({
                    'status': 'completed',
                    'schedule_id': schedule_id,
                    'end_time': time.time()
                })
                
                print(f"Successfully processed file: {filepath}")
                
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                
                processing_status[request_id].update({
                    'status': 'error',
                    'error': str(e),
                    'end_time': time.time()
                })
        
        # Start background processing
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'request_id': request_id,
            'message': 'File uploaded and processing started',
            'check_status_url': f'/api/status/{request_id}'
        })
        
    except Exception as e:
        error_msg = f"Error processing upload: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/status/<request_id>', methods=['GET'])
def check_status(request_id):
    """Check the status of a processing request"""
    if request_id not in processing_status:
        return jsonify({'error': 'Invalid request ID'}), 404
    
    status = processing_status[request_id].copy()
    
    # Calculate processing time
    status['processing_time'] = (status.get('end_time', time.time()) - status['start_time']) * 1000  # in ms
    
    # Include schedule data if processing is complete
    if status['status'] == 'completed' and 'schedule_id' in status:
        schedule_path = os.path.join(SCHEDULES_FOLDER, f"{status['schedule_id']}.json")
        if os.path.exists(schedule_path):
            with open(schedule_path, 'r') as f:
                status['schedule'] = json.load(f)
    
    # Remove internal fields
    for field in ['start_time', 'end_time']:
        status.pop(field, None)
    
    return jsonify(status)

@app.route('/api/schedule/<schedule_id>', methods=['GET'])
def get_schedule_api(schedule_id):
    """Get a generated schedule by ID"""
    schedule_path = os.path.join(SCHEDULES_FOLDER, f"schedule_{schedule_id}.json")
    if not os.path.exists(schedule_path):
        return jsonify({'error': 'Schedule not found'}), 404
    
    with open(schedule_path, 'r') as f:
        schedule_data = json.load(f)
    
    return jsonify(schedule_data)

@app.route('/schedule/<schedule_id>')
def view_schedule(schedule_id):
    """Display the generated schedule in the web interface"""
    schedule_path = os.path.join(SCHEDULES_FOLDER, f"schedule_{schedule_id}.json")
    if not os.path.exists(schedule_path):
        return "Schedule not found", 404
    
    with open(schedule_path, 'r') as f:
        schedule_data = json.load(f)
    
    # Add some default values if they don't exist
    schedule_data.setdefault('generated_at', datetime.now().isoformat())
    schedule_data.setdefault('practice_test_schedule', [])
    schedule_data.setdefault('recommended_resources', [])
    
    return render_template('schedule.html', schedule=schedule_data)

@app.route('/answer')
def answer():
    """Serve the answer page"""
    schedule_id = request.args.get('schedule_id')
    if not schedule_id:
        return "Missing schedule ID", 400
    
    # Check if the schedule file exists in the schedules folder
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    schedules_dir = os.path.join(backend_dir, 'schedules')
    schedule_path = os.path.join(schedules_dir, f"schedule_{schedule_id}.json")
    if not os.path.exists(schedule_path):
        return "Schedule not found", 404
    
    return send_from_directory('../frontend', 'answer.html')

@app.route('/waiting')
def serve_waiting():
    """Serve the waiting page"""
    request_id = request.args.get('request_id')
    if not request_id or request_id not in processing_status:
        return redirect(url_for('index'))
    
    # Check if waiting.html exists, if not return a simple waiting page
    waiting_page = os.path.join('..', 'frontend', 'waiting.html')
    if os.path.exists(waiting_page):
        return send_from_directory('../frontend', 'waiting.html')
    
    # Simple waiting page if waiting.html doesn't exist
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Processing Your Request</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
            .spinner {{ 
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
                margin: 20px auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <h1>Processing Your Request</h1>
        <div class="spinner"></div>
        <p>Please wait while we process your file...</p>
        <p>Request ID: {request_id}</p>
        <script>
            // Auto-refresh the page every 3 seconds to check status
            setTimeout(function() {{
                window.location.reload();
            }}, 3000);
        </script>
    </body>
    </html>
    """

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
            print("Starting file upload and processing...")  # Debug log
            
            # Get client IP address for filename
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ['REMOTE_ADDR'])
            # Clean IP address to be filesystem-safe
            safe_ip = "".join(c if c.isalnum() else "_" for c in client_ip)
            
            # Ensure the upload directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Save the PNG file
            new_filename = f"{safe_ip}_{int(time.time())}.png"  # Add timestamp for uniqueness
            filepath = os.path.join(UPLOAD_FOLDER, new_filename)
            
            print(f"Saving file to: {filepath}")  # Debug log
            
            # Save the PNG file directly
            file.save(filepath)
            
            print("File saved, starting image processing...")  # Debug log
            
            # Process the image with OpenAI's Vision API
            result = process_image_with_openai(filepath)
            
            # If there was an error processing the image
            if not result.get('success'):
                print(f"Error processing image: {result['error']}")  # Debug log
                return jsonify({
                    'status': 'error',
                    'message': 'File uploaded but could not be processed',
                    'error': result.get('error', 'Unknown error occurred'),
                    'filename': new_filename
                }), 400
            
            print("Schedule generated successfully")  # Debug log
            
            # Generate a unique ID for this schedule
            schedule_id = str(uuid.uuid4())
            
            # Ensure the schedules directory exists
            os.makedirs(SCHEDULES_FOLDER, exist_ok=True)
            
            # Save the schedule data to the schedules folder
            schedule_file = os.path.join(SCHEDULES_FOLDER, f'schedule_{schedule_id}.json')
            try:
                with open(schedule_file, 'w', encoding='utf-8') as f:
                    json.dump(result['schedule'], f, ensure_ascii=False, indent=2)
                print(f"Schedule saved to: {os.path.abspath(schedule_file)}")  # Debug log
            except Exception as e:
                print(f"Error saving schedule: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'error': f'Error saving schedule: {str(e)}'
                }), 500
            
            # Return success response with schedule ID and redirect URL
            return jsonify({
                'status': 'completed',
                'message': 'Schedule generated successfully',
                'schedule_id': schedule_id,
                'redirect_url': f'/answer?schedule_id={schedule_id}'
            }), 200
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")  # Debug log
            import traceback
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'error': f'Error processing file: {str(e)}',
                'details': str(e)
            }), 500
    
    return jsonify({
        'status': 'error',
        'error': 'File type not allowed. Please upload a PDF, PNG, JPG, or JPEG file.'
    }), 400

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

@app.route('/api/process_saved_image/<filename>', methods=['POST'])
def process_saved_image(filename):
    """Process an image from the recorded_scores directory"""
    try:
        # Construct the full file path
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        # Generate a request ID
        request_id = str(uuid.uuid4())
        
        # Initialize processing status
        processing_status[request_id] = {
            'status': 'processing',
            'start_time': time.time(),
            'file_path': filepath,
            'schedule_id': None,
            'error': None
        }
        
        # Process the file in the background
        def process_in_background():
            try:
                print(f"Processing saved file: {filepath}")
                result = process_image_with_openai(filepath)
                
                if not result['success']:
                    raise Exception(result.get('error', 'Failed to process file'))
                
                # Save the schedule
                schedule_id = str(uuid.uuid4())
                schedule_path = os.path.join(SCHEDULES_FOLDER, f"{schedule_id}.json")
                with open(schedule_path, 'w') as f:
                    json.dump(result['schedule'], f, indent=2)
                
                # Update processing status
                processing_status[request_id].update({
                    'status': 'completed',
                    'schedule_id': schedule_id,
                    'end_time': time.time()
                })
                
                print(f"Successfully processed file: {filepath}")
                
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                
                processing_status[request_id].update({
                    'status': 'error',
                    'error': str(e),
                    'end_time': time.time()
                })
        
        # Start background processing
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'request_id': request_id,
            'message': 'Image processing started',
            'check_status_url': f'/api/status/{request_id}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list_saved_images', methods=['GET'])
def list_saved_images():
    """List all images in the recorded_scores directory"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                files.append({
                    'name': filename,
                    'size': os.path.getsize(filepath),
                    'modified': time.ctime(os.path.getmtime(filepath))
                })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/schedule/<schedule_id>', methods=['GET'])
def get_schedule(schedule_id):
    """Get a generated schedule by ID in the format expected by the frontend"""
    try:
        print(f"\n=== Processing request for schedule_id: {schedule_id} ===")
        
        # Get the absolute path to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Define the schedules directory path (in project root)
        schedules_dir = os.path.join(project_root, 'schedules')
        
        # Create schedules directory if it doesn't exist
        os.makedirs(schedules_dir, exist_ok=True)
        
        # Construct the full path to the schedule file
        schedule_file = os.path.join(schedules_dir, f'schedule_{schedule_id}.json')
        
        # Debug information
        print(f"Project root: {project_root}")
        print(f"Schedules directory: {schedules_dir}")
        print(f"Schedule file path: {schedule_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory exists: {os.path.exists(schedules_dir)}")
        
        # Check if the file exists
        if not os.path.exists(schedule_file):
            print(f"Schedule file not found: {schedule_file}")
            return jsonify({'error': 'Schedule not found'}), 404
        
        # Read the schedule data
        print(f"Reading schedule file: {schedule_file}")
        with open(schedule_file, 'r', encoding='utf-8') as f:
            schedule_data = json.load(f)
        print(f"Successfully loaded schedule data")
        
        # Create a summary from the student info
        try:
            student_info = schedule_data.get('student_info', {})
            print(f"Student info: {json.dumps(student_info, indent=2)}")
            
            current_scores = student_info.get('current_scores', {})
            target_scores = student_info.get('target_scores', {})
            test_date = student_info.get('test_date', 'Not specified')
            
            summary = f"""
            <h4>Your Study Plan Summary</h4>
            <p><strong>Current Scores:</strong> Math: {current_scores.get('Math', 'N/A')}, 
            Reading: {current_scores.get('Reading', 'N/A')}, 
            Writing: {current_scores.get('Writing', 'N/A')}
            </p>
            <p><strong>Target Test Date:</strong> {test_date}</p>
            <p>This personalized study plan is designed to help you improve your SAT scores with focused practice and review.</p>
            """
        except Exception as e:
            print(f"Error creating summary: {str(e)}")
            summary = "<h4>Your Personalized SAT Study Plan</h4><p>Here's your customized study schedule to help you prepare for the SAT.</p>"
        
        # Transform the weekly schedule data
        weeks = []
        try:
            study_plan = schedule_data.get('study_plan', {})
            print(f"Study plan keys: {list(study_plan.keys())}")
            
            # Ensure study_plan is a dictionary
            if not isinstance(study_plan, dict):
                print(f"Warning: study_plan is not a dictionary: {type(study_plan)}")
                study_plan = {}
            
            # Sort week keys to maintain order (week1, week2, etc.)
            week_keys = [k for k in study_plan.keys() if isinstance(k, str) and k.startswith('week')]
            week_keys.sort()
            
            for week_key in week_keys:
                try:
                    week_data = study_plan[week_key]
                    if not isinstance(week_data, dict):
                        print(f"Warning: {week_key} data is not a dictionary: {type(week_data)}")
                        continue
                        
                    # Get focus areas
                    focus_areas = []
                    if 'focus_areas' in week_data and isinstance(week_data['focus_areas'], list):
                        focus_areas = week_data['focus_areas']
                    
                    # Get all unique topics from daily schedule
                    all_topics = set()
                    daily_schedule = week_data.get('daily_schedule', {})
                    if isinstance(daily_schedule, dict):
                        for day, activities in daily_schedule.items():
                            if isinstance(activities, list):
                                all_topics.update([str(a) for a in activities if a])
                    
                    # Get practice tests for this week
                    practice_tests = []
                    test_schedule = schedule_data.get('practice_test_schedule', [])
                    if isinstance(test_schedule, list):
                        for test in test_schedule:
                            if test and week_key.lower() in str(test).lower():
                                test_desc = test.split(' - ')[1] if ' - ' in test else str(test)
                                practice_tests.append(test_desc)
                    
                    # Add week data
                    weeks.append({
                        'focus_area': ', '.join(str(f) for f in focus_areas) if focus_areas else 'General Study',
                        'topics': list(all_topics)[:5],  # Limit to top 5 unique topics
                        'practice_tests': ', '.join(practice_tests) if practice_tests else 'None',
                        'study_hours': '5-10'  # Default estimate
                    })
                    
                except Exception as e:
                    print(f"Error processing {week_key}: {str(e)}")
                    continue
                    
            print(f"Processed {len(weeks)} weeks of schedule data")
            
        except Exception as e:
            print(f"Error transforming schedule data: {str(e)}")
            # Provide some default data so the page still loads
            weeks = [{
                'focus_area': 'General Study',
                'topics': ['Reading comprehension', 'Math fundamentals', 'Writing skills'],
                'practice_tests': 'Full-length practice test',
                'study_hours': '5-10'
            }]
        
        response_data = {
            'summary': summary,
            'weeks': weeks
        }
            
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
