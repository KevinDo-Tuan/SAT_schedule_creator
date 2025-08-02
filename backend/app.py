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
import socket
import uuid
import time
import glob
from functools import wraps
import pandas as pd

client = OpenAI()

# Function to create a file with the Files API
def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# Getting the file ID
file_id = create_file("path_to_your_image.jpg")

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {
                "type": "input_image",
                "file_id": "/recorded_scores",
            },
        ],
    }],
)

print(response.output_text)
    

@app.route('/')
def index():
    """Serve the main application page"""
    return app.send_static_file('upload.html')

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    print("\n=== New File Upload Request ===")
    print(f"Files received: {request.files}")
    
    if 'file' not in request.files:
        error_msg = 'No file part in the request'
        print(f"Error: {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    file = request.files['file']
    if file.filename == '':
        error_msg = 'No selected file'
        print(f"Error: {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    if not allowed_file(file.filename):
        error_msg = 'File type not allowed. Please upload only a PNG, JPG, or PDF file.'
        print(f"Error: {error_msg}")
        return jsonify({'error': error_msg}), 400
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(SCHEDULES_FOLDER, exist_ok=True)
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Schedules folder: {os.path.abspath(SCHEDULES_FOLDER)}")
    
    try:
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        print(f"Generated request ID: {request_id}")
        
        # Save the uploaded file with a unique name
        filename = secure_filename(f"{request_id}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath):
            error_msg = 'Failed to save uploaded file'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 500
            
        print(f"File saved successfully: {os.path.getsize(filepath)} bytes")
        
        # Store processing status
        processing_status[request_id] = {
            'status': 'processing',
            'start_time': time.time(),
            'file_path': filepath,  # Use the actual saved file path
            'schedule_id': None,
            'error': None
        }
        
        # Start background processing
        def process_in_background():
            try:
                print(f"\n=== Processing file in background (request_id: {request_id}) ===")
                print(f"Processing file: {filepath}")
                
                # Process the file
                result = process_image_to_text(filepath)
                print(f"Processed text (first 100 chars): {str(result)[:100]}...")
                
                if 'error' in result:
                    error_msg = f"Error processing file: {result['error']}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                # Generate SAT schedule
                print("Generating schedule...")
                schedule_result = generate_sat_schedule(result.get('text', ''))
                
                if not schedule_result or 'error' in schedule_result:
                    error_msg = schedule_result.get('error', 'Failed to generate schedule')
                    print(f"Error generating schedule: {error_msg}")
                    raise Exception(error_msg)
                
                # Save the schedule
                print("Saving schedule...")
                schedule_id = save_schedule(schedule_result.get('schedule', {}))
                
                if not schedule_id:
                    error_msg = 'Failed to save schedule - no schedule ID returned'
                    print(error_msg)
                    raise Exception(error_msg)
                
                print(f"Schedule saved with ID: {schedule_id}")
                
                # Update processing status
                processing_status[request_id].update({
                    'status': 'completed',
                    'schedule_id': schedule_id,
                    'end_time': time.time(),
                    'schedule_path': get_schedule_path(schedule_id)
                })
                
                print(f"Background processing completed for request_id: {request_id}")
                
            except Exception as e:
                error_msg = f"Error in background processing: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                
                processing_status[request_id].update({
                    'status': 'error',
                    'error': str(e),
                    'end_time': time.time()
                })
        
        # Start the background task
        print("Starting background processing thread...")
        import threading
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        # Return the request ID to check status later
        response = {
            'request_id': request_id,
            'status': 'processing',
            'message': 'Your file is being processed. Please wait...',
            'check_status_url': f'/api/status/{request_id}'
        }
        
        print(f"Returning response: {response}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

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
            print("Starting file upload and processing...")  # Debug log
            
            # Get client IP address for filename
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ['REMOTE_ADDR'])
            # Clean IP address to be filesystem-safe
            safe_ip = "".join(c if c.isalnum() else "_" for c in client_ip)
            
            # Ensure the upload directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Always save as PDF
            new_filename = f"{safe_ip}_{int(time.time())}.pdf"  # Add timestamp for uniqueness
            filepath = os.path.join(UPLOAD_FOLDER, new_filename)
            
            print(f"Saving file to: {filepath}")  # Debug log
            
            # If the uploaded file is a PDF, save it directly
            if file_ext == 'pdf':
                file.save(filepath)
            # Otherwise, convert it to PDF
            elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
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
                # For other allowed file types, just save as is
                file.save(filepath)
            
            print("File saved, starting text extraction...")  # Debug log
            
            # Process the image with OCR and Gemini
            result = process_image_to_text(filepath)
            
            # If there was an error processing the image
            if 'error' in result:
                print(f"Error processing image: {result['error']}")  # Debug log
                return jsonify({
                    'status': 'error',
                    'message': 'File uploaded but could not be processed',
                    'error': result['error'],
                    'filename': new_filename
                }), 400
            
            print("Text extracted, generating schedule...")  # Debug log
            
            # Generate the schedule using the extracted text
            schedule_result = generate_schedule(result.get('text', ''))
            
            if schedule_result.get('status') != 'success':
                print(f"Error generating schedule: {schedule_result.get('error')}")  # Debug log
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to generate schedule',
                    'error': schedule_result.get('error', 'Unknown error')
                }), 500
            
            print(f"Schedule generated with ID: {schedule_result.get('schedule_id')}")  # Debug log
            
            # Return success response with schedule ID and redirect URL
            return jsonify({
                'status': 'completed',
                'message': 'Schedule generated successfully',
                'schedule_id': schedule_result.get('schedule_id'),
                'redirect_url': f'/answer?schedule_id={schedule_result.get("schedule_id")}'
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

@app.route('/answer')
def answer():
    """Serve the answer page"""
    return send_from_directory('../frontend', 'answer.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
