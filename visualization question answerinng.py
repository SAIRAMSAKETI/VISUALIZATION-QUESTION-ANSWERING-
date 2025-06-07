import os
import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
import speech_recognition as sr
import pyttsx3

def capture_and_save_image(filename="webcam_capture.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam!")
        return False

    print("Press 'q' to capture image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            continue

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to capture
            cv2.imwrite(filename, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Image captured and saved as {filename}")
    return True

def analyze_image(image_path, question):
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        return "Error loading model or processor"

    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, question, return_tensors="pt")
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        print(f"Error processing the image: {e}")
        return "Error processing the image"

def speech_to_text(timeout=5, phrase_time_limit=10):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening for your question...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return None

    try:
        question = recognizer.recognize_google(audio)
        print(f"You asked: {question}")
        return question
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; check your network connection. {e}")
        return None

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    filename = "webcam_capture.jpg"
    if not capture_and_save_image(filename):
        return

    print("Do you want to ask questions in text or speech mode? (Enter 'text' or 'speech')")
    mode = input().strip().lower()
    if mode not in ['text', 'speech']:
        print("Invalid mode selected. Exiting.")
        return

    while True:
        if mode == 'text':
            question = input("Please type your question about the image (or type 'exit' to quit): ").strip()
        else:
            print("Please ask a question about the image (or say 'exit' to quit):")
            question = speech_to_text()

        if question is None:
            continue
        if question.lower() == 'exit':
            break

        answer = analyze_image(filename, question)
        if not answer or "error" in answer.lower():
            answer = "Your asked question is not answered. Is there anything else that I can help with?"

        print(f"Answer: {answer}")
        text_to_speech(f"The answer is: {answer}")

if _name_ == "_main_":
    # Disable GPU usage and oneDNN optimizations
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    main()