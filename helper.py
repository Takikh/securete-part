import cv2
import face_recognition
from ultralytics import YOLO
import os
import numpy as np
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import csv
from datetime import datetime
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Configuration
Gemini_model = "gemini-2.0-flash"
Model_Path = os.getenv("MODEL_PATH", "/home/rebbouh/Desktop/yolo/yolov8n-face.pt")
def log_event_with_gemini(image_path, API_key):
    """Generate a detailed description of an image using the Gemini API."""
    genai.configure(api_key=API_key)
    
    try:
        # Open and preprocess the image
        img = Image.open(image_path)
        img = img.resize((1024, 1024))  # Resize to match the model's expected input
        
        # Initialize Gemini model
        model = genai.GenerativeModel(model_name=Gemini_model)
        
        # Enhanced prompt
        prompt = (
            "Analyze the image of a secured room and provide a detailed description. "
            "Include the following information:\n"
            "- What the person is doing.\n"
            "- Any suspicious or unusual activity.\n"
            "- The type of objects visible in the room.\n"
            "- The overall situation in one concise sentence."
            "-write as one text with out doing a new line and with any special char"
        )
        
        # Generate response
        response = model.generate_content(
            [prompt, img],
            safety_settings={
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
            }
        )
        
        # Validate and clean the response
        if response.candidates:
            candidate_content = response.candidates[0].content
            generated_text = candidate_content.parts[0].text
            cleaned_text = generated_text.strip().replace("'", '"')
            
            # Check if the response is meaningful
            if not cleaned_text or "no description" in cleaned_text.lower():
                return "AI could not generate a meaningful description."
            
            return cleaned_text
    except Exception as e:
        print(f"Error processing the image with Gemini API: {e}")
    
    return "No description generated"

    
def log_to_csv(timestamp, status, description):
    """Log an event to CSV file"""
    # Create CSV if it doesn't exist
    url = 'http://localhost:8000/logs/logs/'  # change port if needed

    data = {
        'description': f'{description}'
    }

    response = requests.post(url, json=data)


def monitor_unauthorized_person(frame, API_key, recording_dir="recordings", capture_dir="captures"):
    """Monitor an unauthorized person with periodic captures and labeled recording"""
    # Create directories if they don't exist
    os.makedirs(recording_dir, exist_ok=True)
    os.makedirs(capture_dir, exist_ok=True)
    
    # Start recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(recording_dir, f"intrusion_{timestamp}.mp4")
    
    # Get video properties
    height, width = frame.shape[:2]
    fps = 20.0
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Time tracking for periodic captures
    start_time = time.time()
    last_capture_time = start_time
    capture_interval = 15  # seconds between captures
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize Gemini response for display
    current_description = "Monitoring unauthorized person..."
    
    print(f"🔴 Recording started: {video_path}")
    print("⚠️ Unauthorized person detected! Monitoring in progress...")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create a copy for YOLO processing
            detection_frame = frame.copy()
            
            # Run YOLO detection on each frame to keep tracking faces
            model = YOLO(Model_Path)
            results = model(detection_frame)
            
            # Add timestamp overlay
            current_time = time.time()
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp_str, (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add security label
            cv2.putText(frame, "SECURITY RECORDING", (width - 230, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Status indicator - Red alert banner at top
            cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 180), -1)
            cv2.putText(frame, "⚠️ UNAUTHORIZED PERSON DETECTED", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add elapsed time
            elapsed = int(current_time - start_time)
            cv2.putText(frame, f"Elapsed: {elapsed}s", (width - 150, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
            # Add the last Gemini description
            # Create text bg for better readability
            cv2.rectangle(frame, (0, height - 80), (width, height - 50), (0, 0, 0), -1)
            cv2.putText(frame, f"AI: {current_description}", (10, height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process any faces detected in this frame
            if hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # Draw red box around detected face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "UNAUTHORIZED", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Record the labeled/annotated frame
            video_writer.write(frame)
            
            # Check if it's time to capture an image for Gemini
            if current_time - last_capture_time >= capture_interval:
                # Generate timestamp for this capture
                capture_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save capture (original frame without labels for better AI analysis)
                img_path = os.path.join(capture_dir, f"capture_{capture_timestamp}.jpg")
                cv2.imwrite(img_path, detection_frame)  # Save the clean frame
                
                # Process with Gemini
                description = log_event_with_gemini(img_path, API_key)
                current_description = description  # Update the displayed description
                
                # Log to CSV
                log_to_csv(capture_timestamp, "Unauthorized Activity", description)
                
                # Update last capture time
                last_capture_time = current_time
                
                print(f"📸 Captured frame at {capture_timestamp}: {description}")
            
            # Display monitoring feed
            cv2.imshow("Security Monitoring", frame)
            
            # Exit on 'q' key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print("🛑 Monitoring stopped")

def detect_faces(yolo_model_path, API_key, allowed_dir="allowed_people", not_allowed_dir="not_allowed"):
    """Detect faces and handle allowed/unauthorized people"""
    # Load YOLO face detection model
    model = YOLO(yolo_model_path)

    # Ensure directories exist
    os.makedirs(allowed_dir, exist_ok=True)
    os.makedirs(not_allowed_dir, exist_ok=True)

    # Preload allowed face encodings
    allowed_face_encodings = []
    allowed_face_names = []
    allowed_face_files = os.listdir(allowed_dir)

    print(f"Loading {len(allowed_face_files)} authorized faces...")
    for filename in allowed_face_files:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(allowed_dir, filename)
        
        try:
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                allowed_face_encodings.append(encodings[0])
                allowed_face_names.append(name)
                print(f"✅ Loaded authorized face: {name}")
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")

    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Counter for non-allowed faces
    not_allowed_id = len([f for f in os.listdir(not_allowed_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print("🔍 Starting face detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        # Create a copy for display
        display_frame = frame.copy()
        
        # Add status indicator
        height, width = display_frame.shape[:2]
        cv2.putText(display_frame, "Scanning for faces...", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Run inference using YOLO for face detection
        results = model(frame)
        
        # Check if we have detection boxes
        if hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
            
            # Process each detected face
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Ensure box coordinates are valid
                if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0] or x2 <= x1 or y2 <= y1:
                    continue
                    
                # Crop the face region
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                # Convert to RGB (required by face_recognition)
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # Get the encoding of the detected face
                encodings = face_recognition.face_encodings(face_rgb)
                
                if encodings:
                    face_enc = encodings[0]
                    
                    # Compare with allowed faces
                    matches = face_recognition.compare_faces(allowed_face_encodings, face_enc, tolerance=0.6)
                    
                    if any(matches):
                        # Found an allowed person
                        match_index = matches.index(True)
                        name = allowed_face_names[match_index]
                        
                        # Draw green box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Allowed: {name}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Add status banner
                        cv2.rectangle(display_frame, (0, 0), (width, 40), (0, 128, 0), -1)
                        cv2.putText(display_frame, "✅ AUTHORIZED PERSON DETECTED", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Display frame with information
                        cv2.imshow("Security System", display_frame)
                        cv2.waitKey(15000)  # Show for 1.5 seconds
                        
                        # Log the authorized access
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_to_csv(timestamp, "Authorized Access", f"Authorized person {name} detected")
                        
                        # Clean up and exit
                        cap.release()
                        cv2.destroyAllWindows()
                        print(f"✅ Authorized person detected: {name}")
                        return "authorized"
                    else:
                        # Unauthorized person - save face
                        face_path = os.path.join(not_allowed_dir, f"unauthorized_{not_allowed_id}.jpg")
                        cv2.imwrite(face_path, face_crop)
                        not_allowed_id += 1
                        
                        # Draw red box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(display_frame, "UNAUTHORIZED", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Add alert banner
                        cv2.rectangle(display_frame, (0, 0), (width, 40), (0, 0, 180), -1)
                        cv2.putText(display_frame, "⚠️ UNAUTHORIZED PERSON DETECTED", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Display frame with information
                        cv2.imshow("Security System", display_frame)
                        cv2.waitKey(500)  # Show for 1.5 seconds
                        
                        # Log the initial unauthorized detection
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_to_csv(timestamp, "Unauthorized Detection", "Unauthorized person detected")
                        
                        # Clean up
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        print(f"⚠️ Unauthorized person detected! Face saved: {face_path}")
                        return "unauthorized"
        
        # Display the frame with scanning status
        cv2.imshow("Security System", display_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    return None

# if __name__ == "__main__":
#     # Configuration
#  # Update with your YOLO model path
#     API_key = "AIzaSyBs-aw-lP7CrxAcmK33UX1v3tfCylMAdyk"  # Replace with actual API key
    
#     # Step 1: Detect faces
#     result = detect_faces(Model_Path, API_key)
    
#     # Step 2: If unauthorized person detected, start monitoring
#     if result == "unauthorized":
#         # Get a fresh frame to start with
#         cap = cv2.VideoCapture(0)
#         ret, frame = cap.read()
#         cap.release()
        
#         if ret:
#             # Start monitoring the unauthorized person
#             monitor_unauthorized_person(frame, API_key)
#         else:
#             print("❌ Failed to get initial frame for monitoring")
#     elif result == "authorized":
#         print("✅ Authorized person detected - system relaxed")
#     else:
#         print("⚠️ No definitive detection made")