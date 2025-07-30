from PIL import Image
import pytesseract

path = r"C:\Users\Do Pham Tuan\OneDrive\Pictures\Screenshots 1\Ảnh chụp màn hình 2025-07-19 103440.png"


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = Image.open(path)
text = pytesseract.image_to_string(img, lang='eng')
print(text)


import google.generativeai1 as genai



# Configure Gemini
genai.configure(api_key="AIzaSyC1wn5_K0JWOhOgvHg0lw1UOn9XHBrdSvA")


# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

prompt = "I have text, list for me their overall SAT score, Math score, Reading and writing. Always return a dictionary with the keys 'SAT', 'Math', 'Reading', and 'Writing'. If the information is not available, return None for that key. " 
    
# Persistent chat session
chat = model.start_chat(history=[prompt])



# Get Gemini response
def get_response(message):
    response = chat.send_message(message)
    return response.text




reply = get_response(text)
print(f"Chatbot: {reply}")

