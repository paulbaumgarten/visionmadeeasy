import cv2
import numpy as np
import os, sys, time, json
from PIL import Image

""" 
visionmadeeasy - Paul Baumgarten, 2019
https://github.com/paulbaumgarten/org.pypi.visionmadeeasy

Built on previous work by 
 * Anirban Kar 2017 - https://github.com/thecodacus/Face-Recognition
 * Marcelo Rovai 2018 - https://github.com/Mjrovai/OpenCV-Face-Recognition
"""

### Set package globals
name = "visionmadeeasy"
version = "2019.02.17"
author = "Paul Baumgarten"
author_website = "pbaumgarten.com"

### Checking minimum version numbers of packages
assert "face" in dir(cv2),                          "Please run: pip install opencv-contrib-python"
assert int(np.__version__.split(".")[0]) >= 1,      "Please run: pip install numpy"
assert int(Image.__version__.split(".")[0]) >= 5,   "Please run: pip install pillow"

### Utility functions

def convert_cv2_to_pil( cv2_image ):
    import cv2
    from PIL import Image
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image

def convert_pil_to_cv2( pil_image ):
    import cv2
    import numpy as np
    from PIL import Image
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv2_image

### START OF CLASS: SimpleVision

class VisionMadeEasy():

    def __init__(self, camera_device_id=0, images_folder=".", cascade_file="haarcascade_frontalface_default.xml" ):
        """
        Constructor

        VisionMadeEasy is a wrapper class for OpenCV to do facial detection and recognition.

        You must have a cascade file downloaded and available for this to function.  Some cascades you might like to use are available at:
        https://github.com/opencv/opencv/tree/master/data/haarcascades
        https://github.com/opencv/opencv_contrib/tree/master/modules/face/data/cascades
        The haarcascade_frontalface_default is bundled with this package.

        Must also have read/write access to the working folder, for a training_file, or you can maually override training_file location with the .set_training_file() function.

        Dependencies: opencv-contrib-python, numpy, Pillow

        :param camera_device_id: The ID number of your attached camera, defaults to 0
        :param images_folder: Path to a local folder to store/read images for computer vision. Folder must exist and be writable.
        :param cascade_file: Path and filename to the cascade file you wish to use (see note above)
        """
        self.camera_device_id=camera_device_id
        self.images_folder = images_folder
        if not os.path.exists(images_folder):
            exit("[SimpleVision] ERROR: images folder not found: "+images_folder)
        self.cascade_file = cascade_file
        if not os.path.exists(cascade_file):
            exit("[SimpleVision] ERROR: cascade file not found: "+cascade_file)
        self.cascade = cv2.CascadeClassifier(cascade_file)
        # some defaults
        self.training_file = "training_data.yml" 
        self.flip = False
        self.camera_width = 640
        self.camera_height = 480
        self.min_detect_width = 40
        self.min_detect_height = 40
        self.names_json = "person_names.json"

    def set_camera_device(self, camera_device_id):
        """ Set the camera id number (default: 0) """
        self.camera_device_id=camera_device_id

    def set_camera_resolution(self, width, height):
        if isinstance(width, int) and isinstance(height, int) and width > 0 and width <= 4096 and height > 0 and height <= 4096:
            self.camera_width = width
            self.camera_height = height

    def set_minimum_detection_resolution(self, width, height):
        if isinstance(width, int) and isinstance(height, int) and width > 0 and width <= 4096 and height > 0 and height <= 4096:
            self.min_detect_width = width
            self.min_detect_height = height

    def set_images_folder(self, images_folder):
        """ Set file path to the images folder """
        self.images_folder = images_folder
        if not os.path.exists(images_folder):
            exit("[SimpleVision] ERROR: images folder not found: "+images_folder)

    def set_training_file(self, training_file="training_data.yml" ):
        """ Set file path to the preferred cascade file """
        self.training_file = training_file
        if not os.path.exists(training_file):
            exit("[SimpleVision] ERROR: training_file file not found: "+training_file)

    def set_cascade(self, cascade_file="haarcascade_frontalface_default.xml" ):
        """ Set file path to the preferred cascade file """
        self.cascade_file = cascade_file
        if not os.path.exists(cascade_file):
            exit("[SimpleVision] ERROR: cascade file not found: "+cascade_file)
        self.cascade = cv2.CascadeClassifier(cascade_file)

    def set_flip(self, flip ):
        """ Set to true to vertically flip the image from the camera (usually required for Raspberry Pi) """
        if isinstance(flip, bool):
            self.flip = flip
        else:
            self.flip = False

    ### Internal/private functions

    def __add_record_to_json(self, json_file_name, key, val=None ):
        # If we have a val, append it to the json file
        if val is not None:
            # initialise
            data = {}
            # load existing key/vals from the json file
            try:
                with open(json_file_name, "r") as f:
                    data = json.load(f)
            except: # if the file doesn't exist, no big deal, let's move on
                pass
            # Add/overwrite this to the people dict
            data[key] = val
            # Save back to the json file
            try:
                with open( json_file_name, "w") as f:
                    json.dump(data, f)
            except OSError: # Fatal error
                exit("Error writing to file "+json_file_name)

    def __get_camera(self):
        # cv2.namedWindow("preview") # Mac
        cap = cv2.VideoCapture(self.camera_device_id)
        cap.set(3, self.camera_width)
        cap.set(4, self.camera_height)
        return cap


    ### Public functions

    def detect_face(self, callback ):
        """ 
        Will detect a face in the camera, and call the `callback` function if a face is found detected.

        The callback must accept two paramaters: coordinates and img.
         * `coordinates` is a dict of {x,y,w,h} pixel coordinates for where the face appears inside the image
         * `img` is the OpenCV image object
        The callback must return True or False, to indicate if the function should continue looping.

        :param callback: a callback function see note above.
        :returns: The last matched Open CV image object
        """
        assert callable(callback),             "The callback parameter must be a function"

        cap = self.__get_camera()
        loop = True # loop can be stopped by the callback function
        while loop:
            # Read image from the camera
            ret, img = cap.read()
            assert ret, "Error reading from capture device "+str(self.camera_device_id)
            if self.flip:
                img = cv2.flip(img, -1)
            # Convert image to grey scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect any faces in the image? Put in an array
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,     
                minSize=(self.min_detect_width, self.min_detect_height)
            )
            # For every face we found
            for (x,y,w,h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                # Execute the callback, alerting we found a face!
                if callback is not None:
                    loop = callback({"x":x,"y":y,"w":w,"h":h}, img)
            # Show the face on screen
            cv2.imshow('video',img)
            # Check for exit key press
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        cap.release()
        cv2.destroyAllWindows()
        return img

    def record_face_dataset( self, images_to_record=30, interval=1, person_identifier=1, person_name=None ):
        """
        This function will:
        1. take multiple photos of a person
        2. save each photo to the images_folder
        3. link the persons identifier number to their name in the `person_names.json` in the same folder
        """
        assert isinstance(person_identifier, int), "Person identifier must be an integer, unqiue to the person"

        # Add the person name to our file person_names.json
        self.__add_record_to_json(os.path.join(self.images_folder , self.names_json), person_identifier, person_name)
        cap = self.__get_camera()
        images_recorded = 0
        while images_recorded < images_to_record:
            # Read image from the camera
            ret, img = cap.read()
            assert ret, "Error reading from capture device "+str(self.camera_device_id)
            if self.flip:
                img = cv2.flip(img, -1)
            # Convert image to grey scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect any faces in the image? Put in an array
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,     
                minSize=(self.min_detect_width, self.min_detect_height)
            )
            # For every face we found
            for (x,y,w,h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                # Save the captured image into the datasets folder
                filename = os.path.join(self.images_folder , "person_"+str(person_identifier)+"_"+str(images_recorded) + ".jpg")
                print("Saving {}".format(filename))
                cv2.imwrite(filename, gray[y:y+h,x:x+w])
                # Put count text on the image
                text = str("Captured image "+str(images_recorded)+" of "+str(images_to_record))
                cv2.putText(img, text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                images_recorded += 1               
                # Display the captured image
                cv2.imshow('image', img)
                # Pause for interval number of seconds
                time.sleep(interval)
            # Show the image on screen
            cv2.imshow('image', img)
            # Check for exit key press
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        cap.release()
        cv2.destroyAllWindows()
        return img

    def train_from_faces( self ):
        """
        Will analyse all the faces in the object's images_folder. Depending on the number of images this could take some time (allow 10 seconds per 100 images).
        Updates the object's training_file with the resulting calculations for use `recognise_face` function.
        """
        # Path for face image database
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Find all files in the folder
        imagePaths = [os.path.join(self.images_folder,f) for f in os.listdir(self.images_folder)]     
        faceSamples=[]
        ids = []
        # For all files in the folder
        for imagePath in imagePaths:
            try:
                # Load the image file, convert it to grayscale
                PIL_img = Image.open(imagePath).convert('L')
                # Convert to numpy array
                img_numpy = np.array(PIL_img,'uint8')
                # Get the identifier of this person
                id = int(os.path.split(imagePath)[-1].split("_")[1])
                # Find the face in the numpy array of image data
                faces = self.cascade.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            except OSError:
                pass # Probably not an image file, so ignore it (eg: .DS_Store file or some other stupid indexing)
        # Train with those faces
        recognizer.train(faceSamples, np.array(ids))
        # Save the model into trainer yml data file
        recognizer.write(self.training_file) # recognizer.save() worked on Mac, but not on Pi
        # Return the numer of faces trained
        return len(np.unique(ids))

    def recognise_face( self, callback ):
        """ 
        Will aim to recognise a face in the camera, and call the `callback` function if a face is found detected.

        The callback must accept four paramaters: coordinates, person_name, confidence, img.
         * `coordinates` is a dict of {x,y,w,h} pixel coordinates for where the face appears inside the image
         * `person_name` the name that was provided when the photos were recorded and is stored in `person_names.json`
         * `confidence` a number between 0 and 100 (100 = highest confidence)
         * `img` is the OpenCV image object
        The callback must return True or False, to indicate if the function should continue looping.

        :param callback: a callback function see note above.
        :returns: The last matched Open CV image object
        """
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.training_file)
        # Load the person_names.json file
        names = {}
        with open(os.path.join(self.images_folder , self.names_json), "r") as f:
            names = json.load(f)
        cap = self.__get_camera()
        # Define min window size to be recognized as a face
        loop = True # callback can terminate loop
        while loop:
            # Read image from the camera
            ret, img = cap.read()
            assert ret, "Error reading from capture device "+str(self.camera_device_id)
            if self.flip:
                img = cv2.flip(img, -1)
            # Convert image to grey scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect any faces in the image? Put in an array
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,     
                minSize=(self.min_detect_width, self.min_detect_height)
            )
            # For every face we found
            for (x,y,w,h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                # Recognise the face if possible
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                person = "unknown"
                # If confidence is less then 100, deem the person recognised (0 == perfect match) 
                if (confidence < 100):
                    if str(id) in names:
                        person = names[str(id)]
                    else:
                        person = "person_"+str(id)
                    confidence = round(100 - confidence)
                else:
                    person = "unknown"
                    confidence = round(100 - confidence)
                # Execute the callback, alerting we found a face!
                if callback is not None:
                    loop = callback({"x":x,"y":y,"w":w,"h":h}, person, confidence, img)
                # Put name and confidence rating text on image
                cv2.putText(img, person, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence)+"%", (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)  
            # Show the image on screen
            cv2.imshow('image', img)
            # Check for exit key press
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        cap.release()
        cv2.destroyAllWindows()
        return img

