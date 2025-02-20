import cv2
import numpy as np

def faceBox(faceNet, frame):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frameWidth, x2), min(frameHeight, y2)

            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, bboxs

# Load pre-trained models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageGroups = [2, 6, 12, 20, 32, 43, 53, 100]
genderList = ['Male', 'Female']

# Initialize webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    frame, bboxs = faceBox(faceNet, frame)

    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue  # Skip if no valid face detected

        # Convert face to RGB for better model compatibility
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Prepare face for deep learning model
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # Predict Age with Improved Estimation
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        ageDist = agePred[0] / agePred[0].sum()  # Normalize predictions
        exact_age = int(np.dot(ageDist, ageGroups))  # Weighted sum for better accuracy

        # Display result
        label = "{}: {} years".format(gender, exact_age)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

