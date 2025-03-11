import cv2
import numpy as np

# Load pre-trained MobileNet SSD model and Haar Cascade
prototxt_path = MobileNetSSD_deploy.prototxt  # Path to the MobileNet SSD deploy prototxt file
model_path = MobileNetSSD_deploy.caffemodel  # Path to the MobileNet SSD model file
haar_cascade_path = "haarcascade_frontalface_default.xml"  # Path to Haar Cascade XML file

# Load the models
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
haar_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Start the video capture
cap = cv2.VideoCapture(niggavideo)  # Use webcam as input (change to video file path if needed)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match MobileNet SSD input size
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (300, 300))

    # Prepare MobileNet SSD input
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Draw detections from MobileNet SSD
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Threshold for confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"SSD {confidence:.2f}"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Apply Haar Cascade for additional face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Haar", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the output frame
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
