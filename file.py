import cv2
import numpy as np
import smtplib as smtp
from threading import Thread
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

def send_email(frame):
    ImgFileName = "./detected_img/img.jpg"
    cv2.imwrite(ImgFileName, frame) 

    with open(ImgFileName,'rb') as f:
        img_data = f.read()

    msg = MIMEMultipart()
    msg['Subject'] = 'Person Detected'
    msg['From'] = 'projectuseonly943@gmail.com'
    msg['To'] = 'battelargames@gmail.com'

    text = MIMEText("Person Detected!!!")
    msg.attach(text)

    image = MIMEImage(img_data, name="Detected_obj.jpg")
    msg.attach(image)

    s = smtp.SMTP('smtp.gmail.com', 587)
    # start TLS for security
    s.ehlo()
    s.starttls()
    s.ehlo()
    # Authentication
    s.login("projectuseonly943@gmail.com", "vukx vkta aaey lghx")
    # sending the mail
    s.sendmail(msg['From'], msg['To'], msg.as_string())
    # terminating the session
    s.quit()

# Load pre-trained model and configuration file
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
threads = []
# Open a video file or an image file or a camera stream
cap = cv2.VideoCapture("./niggavideo.mp4")  # Change the index to use a different camera

# List of classes for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

sending_threshold = 60
sending_counter = sending_threshold
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    # Resize the frame to 300x300 pixels and convert to blob
    nw = 16*30
    nh = 9*30
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (nw, nh)), 0.007843, (nw, nh), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.50:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in CLASSES:  # Objects of interest
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            if CLASSES[idx] in ['person']:
                sending_counter -= 1
            
            if sending_counter <= 0:
                threads.append(Thread(target=send_email, args=(frame,)))
                threads[-1].start()
                print("Person detected Email was Sent")
                sending_counter = sending_threshold
                
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    # Break the loop
    if key == ord("q"):
        break

for thread in threads:
    thread.join()

# Release the capture
cap.release()
cv2.destroyAllWindows()
