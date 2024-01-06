import tkinter as tk
from tkinter import Label, Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set the mode based on your requirement (train/display)
mode = "display"

# Load pre-trained weights
if mode == "display":
    weights_path = r'model.h5'
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights(weights_path)

# Create Tkinter GUI window
app = tk.Tk()
app.title("Real-time Emotion Detection")

# Create labels to display emotion
emotion_label = Label(app, text="Detected Emotion:", font=("Helvetica", 16))
emotion_label.pack(pady=10)

# Create canvas to display video feed
canvas = Canvas(app, width=800, height=600)
canvas.pack()

# Function to update the canvas with the webcam feed
def update_canvas(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (800, 600))
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

# Function to update the detected emotion label
def update_emotion_label(emotion):
    emotion_label.config(text=f"Detected Emotion: {emotion}")

# Start the webcam feed and update the GUI
cap = cv2.VideoCapture(0)
# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Find faces and perform emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        detected_emotion = emotion_dict[maxindex]
        cv2.putText(frame, detected_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Update GUI
        update_emotion_label(detected_emotion)

    # Update canvas
    update_canvas(frame)
    app.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ... (rest of your code)


cap.release()
cv2.destroyAllWindows()
app.mainloop()
