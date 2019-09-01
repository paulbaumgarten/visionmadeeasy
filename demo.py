import visionmadeeasy
from PIL import Image
import uuid

### Demonstration of functionality

def i_see_a_face( location, img ):
    print(f"I see a face!!! It is at {location['x']},{location['y']}")
    im = Image.fromarray(img)
    im.save( str(uuid.uuid1())+".jpg")
    return True # must return True to keep the loop alive

def i_recognise_a_face( location, person_name, confidence, img ):
    print(f"Hello {person_name}! I am {confidence}% sure it is you :-)")
    return True # must return True to keep the loop alive

if __name__ == "__main__":
    quit = False
    while not quit:
        print("Demonstration time! Menu of options...")
        print("1. Detect faces")
        print("2. Record faces")
        print("3. Train for faces recorded")
        print("4. Recognise faces (must do training first)")
        print("5. Exit")
        choice = int(input("Enter your option (1 to 5):"))

        vme = visionmadeeasy.VisionMadeEasy(0, "..\\ch.isl.python.facerecognition\\dataset")
        vme.set_training_file("..\\ch.isl.python.facerecognition\\training_data.yml")

        if choice == 1:
            print("[face_vision] Task: Searching for faces.\nLook at the camera! (press ESC to quit)")
            # Demo of detecting faces
            vme.detect_face(i_see_a_face)

        elif choice == 2:
            print("About to save 50 images of different angles etc of a person, saving to folder ./dataset")
            id = int(input("Enter unique person number: "))
            n = input("Enter person name: ")
            print("Smile! :-)")
            # Demo of recording faces
            vme.record_face_dataset(images_to_record=50, interval=1, person_identifier=id, person_name=n)

        elif choice == 3:
            print("[face_vision] Task: Training... please wait...")
            # Demo of training faces
            vme.train_from_faces()

        elif choice == 4:
            print("[face_vision] Task: Searching for faces I recognise.\nLook at the camera! (press ESC to quit)")
            # Demo of recognising faces
            vme.recognise_face(i_recognise_a_face)

        elif choice == 5:
            quit = True

print("Goodbye!")
