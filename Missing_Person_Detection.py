import cv2
import dlib
import bz2
import shutil
import numpy as np

# Input and output file paths for the compressed and decompressed models
input_face_recognition_model_bz2 = "dlib_face_recognition_resnet_model_v1.dat.bz2"
output_face_recognition_model_dat ="dlib_face_recognition_resnet_model_v1.dat"
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
missing_person_image_path = "praj.jpg"


# Decompress the face recognition model .bz2 file
with bz2.open(input_face_recognition_model_bz2, "rb") as source_file, open(output_face_recognition_model_dat, "wb") as dest_file:
    shutil.copyfileobj(source_file, dest_file)

# Load the face recognition model
face_recognizer = dlib.face_recognition_model_v1(output_face_recognition_model_dat)

# Load the shape predictor model
predictor = dlib.shape_predictor(shape_predictor_path)

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Load the missing person's image
missing_person_image = cv2.imread(missing_person_image_path)
missing_person_image = cv2.cvtColor(missing_person_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Compute face descriptor for the missing person's image
missing_person_face_rects = detector(missing_person_image)
missing_person_shape = predictor(missing_person_image, missing_person_face_rects[0])  # Assuming one face is detected
missing_person_face_descriptor = face_recognizer.compute_face_descriptor(missing_person_image, missing_person_shape)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (dlib uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame using dlib face detector
    face_rects = detector(rgb_frame)

    # Iterate over detected faces
    for face_rect in face_rects:
        # Compute face descriptor (a numerical representation of the face)
        shape = predictor(rgb_frame, face_rect)
        face_descriptor = face_recognizer.compute_face_descriptor(rgb_frame, shape)

        # Compare the face descriptor with the missing person's face descriptor
        distance = np.linalg.norm(np.array(face_descriptor) - np.array(missing_person_face_descriptor))

        # If the distance is below a certain threshold, consider it a match
        threshold = 0.4  # You can adjust this threshold value based on your requirement
        if distance < threshold:
            print("Person Found")
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, "Person Found", (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Missing Person Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
