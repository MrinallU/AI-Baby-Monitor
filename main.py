import cv2
import numpy as np
from deepface import DeepFace
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'  # Replace with your SMTP server
SMTP_PORT = 587
SENDER_EMAIL = 'your_email@gmail.com'
SENDER_PASSWORD = 'your_email_password'
RECIPIENT_EMAIL = 'recipient_email@gmail.com'

# Function to send email
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
        server.quit()
        print(f"Email sent successfully to {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to handle mouse events for drawing bounding boxes
def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt, bounding_boxes_too_close, bounding_boxes_left_zone, draw_step, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, top_left_pt, (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw Zones", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        if draw_step == 1:
            bounding_boxes_too_close.append({"coordinates": [top_left_pt, bottom_right_pt]})
        elif draw_step == 2:
            bounding_boxes_left_zone.append({"coordinates": [top_left_pt, bottom_right_pt]})
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2) if draw_step == 1 else cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 0, 255), 2)
        cv2.imshow("Draw Zones", frame)

# Function to analyze emotion using DeepFace
def analyze_emotion(face_image):
    try:
        analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        emotion = analysis['dominant_emotion']
        confidence = analysis['emotion'][emotion]
        return {'emotion': emotion, 'confidence': confidence}
    except Exception as e:
        print(f"Emotion analysis failed: {e}")
        return {'emotion': None, 'confidence': 0}

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
# Handling the difference in OpenCV versions for getUnconnectedOutLayers
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    # OpenCV 4.4 and above return a list of single-element lists
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Define dangerous objects
dangerous_objects = ["gun", "knife", "explosive", "scissors", "medication", "sharp object", "alcohol", "lighter"]

# Open a connection to the webcam (use 0 for default webcam)
cap = cv2.VideoCapture(0)

# Create a window for drawing zones
cv2.namedWindow("Draw Zones")
cv2.setMouseCallback("Draw Zones", draw_rectangle)

# Initialize variables
drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
bounding_boxes_too_close = []
bounding_boxes_left_zone = []
draw_step = 1  # Start with drawing zones where the person should not be too close
frame_copy = None

print("Step 1: Draw zones where the person should not be too close.")
print("Draw the zones by clicking and dragging the mouse. Press 'ESC' to proceed to the next step.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_copy = frame.copy()

    # Draw existing bounding boxes
    if draw_step == 1:
        for bbox in bounding_boxes_too_close:
            cv2.rectangle(frame_copy, tuple(bbox["coordinates"][0]), tuple(bbox["coordinates"][1]), (0, 255, 0), 2)
    elif draw_step == 2:
        for bbox in bounding_boxes_left_zone:
            cv2.rectangle(frame_copy, tuple(bbox["coordinates"][0]), tuple(bbox["coordinates"][1]), (0, 0, 255), 2)

    cv2.imshow("Draw Zones", frame_copy)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key
        if draw_step == 1:
            draw_step = 2  # Move to the next step: drawing zones where the person cannot leave
            print("Step 2: Draw zones where the person cannot leave.")
            print("Draw the zones by clicking and dragging the mouse. Press 'ESC' to finish.")
        else:
            break

# Close the drawing window
cv2.destroyWindow("Draw Zones")

print("Monitoring started. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    person_found = False
    dangerous_object_found = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in dangerous_objects:
                dangerous_object_found = True

            if confidence > 0.5 and classes[class_id] == "person":
                person_found = True
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"Person {i+1}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), font, 1, (0, 255, 0), 2)

            # Extract the face from the frame
            face_y1 = y
            face_y2 = y + int(h / 2)
            face_x1 = x
            face_x2 = x + w
            face_image = frame[face_y1:face_y2, face_x1:face_x2]

            # Analyze the emotion using DeepFace
            emotion_response = analyze_emotion(face_image)

            # Check if the detected emotion indicates distress
            if emotion_response['emotion'] in ['angry', 'sad', 'fear', 'disgust'] and emotion_response['confidence'] > 0.75:
                cv2.putText(frame, "Distress Detected!", (x, y - 10), font, 1, (0, 0, 255), 2)
                print(f"Alert: {label} is in distress (Emotion: {emotion_response['emotion']}, Confidence: {emotion_response['confidence']:.2f})")
                send_email("Distress Detected", f"{label} is in distress (Emotion: {emotion_response['emotion']}, Confidence: {emotion_response['confidence']:.2f})")

            # Check if person is too close to any preset bounding box
            for bbox in bounding_boxes_too_close:
                box_coordinates = bbox["coordinates"]
                corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

                for corner in corners:
                    corner_x, corner_y = corner
                    distance_to_edge_x = min(abs(corner_x - box_coordinates[0][0]), abs(corner_x - box_coordinates[1][0]))
                    distance_to_edge_y = min(abs(corner_y - box_coordinates[0][1]), abs(corner_y - box_coordinates[1][1]))
                    closeness_threshold = 20  # Example threshold value, modify as needed

                    if (
                        distance_to_edge_x < closeness_threshold
                        or distance_to_edge_y < closeness_threshold
                        or cv2.pointPolygonTest(np.array(box_coordinates), (corner_x, corner_y), False) >= 0
                    ):
                        cv2.putText(frame, "Too Close!", (x, y - 40), font, 1, (0, 0, 255), 2)
                        print(f"Alert: {label} is too close to the drawn zone")
                        send_email("Proximity Alert", f"{label} is too close to the drawn zone.")

            # Check if person leaves any preset bounding box
            for bbox in bounding_boxes_left_zone:
                box_coordinates = bbox["coordinates"]
                if not (x > box_coordinates[0][0] and y > box_coordinates[0][1] and (x + w) < box_coordinates[1][0] and (y + h) < box_coordinates[1][1]):
                    cv2.putText(frame, "Left Zone!", (x, y - 70), font, 1, (0, 0, 255), 2)
                    print(f"Alert: {label} left the drawn zone")
                    send_email("Zone Departure Alert", f"{label} left the drawn zone.")

    if not person_found:
        cv2.putText(frame, "Alert: Person not found in the scene", (10, 30), font, 1, (0, 0, 255), 2)
        print("Alert: Person not found in the scene")
        send_email("Person Not Found", "Alert: Person not found in the scene.")
    if dangerous_object_found:
        cv2.putText(frame, "Alert: Dangerous object in the scene", (10, 60), font, 1, (0, 0, 255), 2)
        print("Alert: Dangerous object in the scene")
        send_email("Dangerous Object Alert", "Alert: Dangerous object in the scene.")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
