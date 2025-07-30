import os
from PIL import Image
import pytesseract
import google.generativeai as genai

# Folder containing images
folder_path = r"C:\Users\Do Pham Tuan\OneDrive\Pictures\Screenshots 1" # 1)Create folder to stored image in your backend folder 2) Change this path to 

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Gemini
genai.configure(api_key="AIzaSyC1wn5_K0JWOhOgvHg0lw1UOn9XHBrdSvA")
model = genai.GenerativeModel("gemini-1.5-flash")
promptpy = open("backend/prompt.md", "r").read().strip()
prompt = {
    "role": "user",
    "parts": [promptpy]
}
chat = model.start_chat(history=[prompt])

def get_response(message):
    response = chat.send_message(message)
    return response.text

# Loop through all PNG and JPG files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(folder_path, filename)
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang='eng')
        print(f"Text from {filename}:\n{text}\n")
        reply = get_response(text)
        print(f"Chatbot for {filename}: {reply}\n{'-'*40}\n")




