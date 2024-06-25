import cv2
import dlib
import numpy as np
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

known_faces = []
known_labels = []

def get_face_descriptor(img, rect):
    shape = predictor(img, rect)
    face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)

def load_known_faces(file_paths):
    global known_faces, known_labels
    for file_path in file_paths:
        img = cv2.imread(file_path)
        detected_faces = detector(img, 1)
        if detected_faces:
            known_faces.append(get_face_descriptor(img, detected_faces[0]))
            known_labels.append(file_path.split('/')[-1].split('.')[0])
    known_faces = np.array(known_faces)
    messagebox.showinfo("Info", "Known faces loaded successfully!")

def detect_and_recognize_faces(img):
    detected_faces = detector(img, 1)
    recognized_faces = []

    for rect in detected_faces:
        face_descriptor = get_face_descriptor(img, rect)
        matches = np.linalg.norm(known_faces - face_descriptor, axis=1)
        min_index = np.argmin(matches)
        label = known_labels[min_index] if matches[min_index] < 0.6 else "Unknown"
        recognized_faces.append((rect, label))

    return recognized_faces

def upload_known_faces():
    file_paths = filedialog.askopenfilenames(title="Select Known Face Images", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_paths:
        load_known_faces(file_paths)

def upload_and_recognize():
    file_path = filedialog.askopenfilename(title="Select Image for Recognition", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = cv2.imread(file_path)
        recognized_faces = detect_and_recognize_faces(img)
        for rect, label in recognized_faces:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        display_image(img)

def display_image(img):
    b, g, r = cv2.split(img)
    img = cv2.merge((r, g, b))
    im_pil = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    panel.image = imgtk

# Set up the GUI
root = Tk()
root.title("Face Recognition")

panel = Label(root)
panel.pack(padx=10, pady=10)

btn_load_known = Button(root, text="Upload Known Faces", command=upload_known_faces)
btn_load_known.pack(pady=10)

btn_recognize = Button(root, text="Upload and Recognize", command=upload_and_recognize)
btn_recognize.pack(pady=10)

root.mainloop()
