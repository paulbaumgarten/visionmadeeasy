import cv2
import numpy as np
import os, sys, time, json
from PIL import Image
import uuid

### GLOBAL SETTINGS

TRAINING_FILE = "training_data.yml" 
CAMERA_FLIP = False      # Usually needed for Raspberry Pi camera module
CAMERA_WIDTH = 640
CAMEAR_HEIGHT = 480
MIN_DETECT_WIDTH = 40
MIN_DETECT_HEIGHT = 40
IMAGES_FOLDER = "images"
CASCADE_FILE = "haarcascade_frontalface_default.xml"

def convert_cv2_to_pil( cv2_image ):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def convert_pil_to_cv2( pil_image ):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

class Camera():
    def __init__(self, camera_device_id=0, width=1280, height=720, flip=False ):
        self.camera_device_id=camera_device_id
        self.flip = flip
        self.camera_width = 1280
        self.camera_height = 720
        self.cap = cv2.VideoCapture(self.camera_device_id)
        self.cap.set(3, self.camera_width)
        self.cap.set(4, self.camera_height)

    def record_video(self, length=0.0, filename="", per_frame_callback=None, preview=False ):
        pass

    def record_video_stop(self):
        pass

    def take_photo(self, preview=False):
        # Read image from the camera
        ret, img = self.cap.read()
        if self.flip:
            img = cv2.flip(img, -1)
        if preview:
            cv2.imshow(img)
        # Convert from CV2 image to PIL image
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

def get_faces(img, cascade_file):
    if not os.path.exists(cascade_file):
        raise Exception("[get_faces] Cascade file does not exist")
    if not isinstance(img, Image.Image):
        raise Exception("[get_faces] Not a PIL.Image.Image object")
    # Convert from PIL image to CV2 image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Returns the colour of face, grayscale of face, and full image containing face if there is a face in the photo
    cascade = cv2.CascadeClassifier(cascade_file)
    # Convert image to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect any faces in the image? Put in an array
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(MIN_DETECT_WIDTH, MIN_DETECT_HEIGHT)
    )
    # If there is a face
    return faces


### Demonstration of functionality

def record_faces(person_name):
    # Check the folders exist
    if not os.path.exists("images"):
        os.mkdir("images")
    if not os.path.exists(f"images/{person_name}"):
        os.mkdir(f"images/{person_name}")
    # Open the camera
    camera = Camera(0)
    for i in range(50):
        # Take a photo
        photo = camera.take_photo()
        # Any faces in the photo?
        faces = get_faces(photo, CASCADE_FILE)
        if faces is not None and len(faces) > 0:
            for (x,y,w,h) in faces:
                # Extract the face portion of the image
                face_image = photo.crop((x,y,x+w,y+h))
                # Save the face image
                filename = f"images/{person_name}/{i}.jpg"
                face_image.save(filename, "jpeg")
                print(i)

def train_from_faces(training_file):
    """
    Will analyse all the faces in the object's images folder. 
    Depending on the number of images this could take some time (allow 10 seconds per 100 images).
    Updates the object's training_file with the resulting calculations for use `recognise_face` function.
    """
    # Path for face image database
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Find all people to train on
    folders = os.listdir("images")
    # Delete any non-folders
    folders = [f for f in folders if os.path.isdir("images/"+f)]
    # Create our data storage lists
    face_images = []
    face_numbers = []
    face_number = 0
    for folder in folders:
        print(f"Processing {folder}...")
        for i in range(50):
            # Load the image file, convert it to grayscale
            pimage = Image.open(f"images/{folder}/{i}.jpg").convert('L')
            # Convert to numpy array
            nimage = np.array(pimage,'uint8')
            face_images.append(nimage)
            face_numbers.append(face_number)   # The folder should be named for the person
        face_number += 1
    # Train with those faces
    recognizer.train(face_images, np.array(face_numbers))
    # Save the model into trainer yml data file
    recognizer.write(training_file) # recognizer.save() worked on Mac, but not on Pi
    # Return the numer of faces trained
    return len(np.unique(face_numbers))

def recognise_faces(training_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(training_file)
    # Find all folders we trained on
    folders = os.listdir("images")
    # Delete any non-folders
    folders = [f for f in folders if os.path.isdir("images/"+f)]
    # Take a photo
    camera = Camera(0)
    photo = camera.take_photo()
    # Any faces in the photo?
    faces = get_faces(photo, CASCADE_FILE)
    if faces is not None and len(faces) > 0:
        for (x,y,w,h) in faces:
            # Extract the face portion of the image
            face_image = photo.crop((x,y,x+w,y+h))
            gray = cv2.cvtColor(convert_pil_to_cv2(face_image), cv2.COLOR_BGR2GRAY)
            id, confidence = recognizer.predict(gray)
            # If confidence is less then 100, deem the person recognised (0 == perfect match) 
            if (confidence < 100):
                person = folders[id]
            else:
                person = "unknown"
            confidence = round(100 - confidence)
            print(f"I see {person} with confidence of {confidence}%\n")
    else:
        print(f"No faces detected\n")

if __name__ == "__main__":
    choice = ""
    while choice != "X":
        print("Demonstration time! Menu of options...")
        print("1. Record faces")
        print("2. Train for faces recorded")
        print("3. Recognise faces (must do training first)")
        print("X. Exit")
        choice = ""
        while choice not in ["1", "2", "3", "X"]:
            choice = input("Enter your option...")

        if choice == "1":
            print("About to save 50 images of different angles etc of a person")
            name = input("Enter person name: ")
            name = name.replace(" ","_")
            print("Smile! :-)")
            record_faces(name)

        elif choice == "2":
            print("Studying your faces... please wait...")
            train_from_faces(TRAINING_FILE)

        elif choice == "3":
            print("Let's see if I recognise you...")
            recognise_faces(TRAINING_FILE)

print("Goodbye!")
