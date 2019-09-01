# Vision Made Easy

This project aims to remove a lot of the complexity of dealing with the Open CV for beginner level programmers to experiment with face detection and recognition.

This project has been initially developed for use within my own classes that I teach but I hope it might find use for others too.

## PROJECT HOME

* [VisionMadeEasy](https://pbaumgarten.com/visionmadeeasy)

## INSTALL

```
pip install visionmadeeasy
```

To successfully run the demo, you will also have to...

* Download a cascade file such as `haarcascade_frontalface_default.xml` from [https://github.com/opencv/opencv/tree/master/data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades) and save it into your project folder
* Create a sub-folder called "datasets" in your project folder. This is where it will store your training photos.
* Make sure you have a web camera attached :-)

## DEMO CODE

```python
import visionmadeeasy

def i_see_a_face( location, img ):
    print(f"I see a face!!! It is at {location['x']},{location['y']}")
    return True # must return True to keep the loop alive

def i_recognise_a_face( location, person_name, confidence, img ):
    print(f"Hello {person_name}! I am {confidence}% sure it is you :-)")
    return True # must return True to keep the loop alive

if __name__ == "__main__":
    vme = visionmadeeasy.VisionMadeEasy(0, "dataset")
    quit = False
    while not quit:
        print("Demonstration time! Menu of options...")
        print("1. Detect faces")
        print("2. Record faces")
        print("3. Train for faces recorded")
        print("4. Recognise faces (must do training first)")
        print("5. Exit")
        choice = int(input("Enter your option (1 to 5):"))

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
```

## AUTHOR

* [Paul Baumgarten](https://pbaumgarten.com/)

## LICENSE

MIT License (C) 2019 Paul Baumgarten

