import face_recognition
import cv2
import numpy as np

json_filepath = "./json/"
video_name = "ShortTC-TG"
video = "./test_video/ShortTC-TG.mp4"
video_capture = cv2.VideoCapture(video)

frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv2.VideoWriter('./outputs/facesTG.mp4', fourcc, 15.0, (frame_width, frame_height))
# Load a sample picture and learn how to recognize it.
tom_image = face_recognition.load_image_file("./image/tom.jpg")
tom_face_encoding = face_recognition.face_encodings(tom_image)[0]

# Load a second sample picture and learn how to recognize it.
anthony_image = face_recognition.load_image_file("./image/anthony.jpg")
anthony_face_encoding = face_recognition.face_encodings(anthony_image)[0]

# Load a third sample picture and learn how to recognize it.
val_image = face_recognition.load_image_file("./image/val.jpg")
val_face_encoding = face_recognition.face_encodings(val_image)[0]

# Load a fourth sample picture and learn how to recognize it.
skerritt_image = face_recognition.load_image_file("./image/skerritt.jpg")
skerritt_face_encoding = face_recognition.face_encodings(skerritt_image)[0]
# Create arrays of known face encodings and their names
known_face_encodings = [
    tom_face_encoding,
    anthony_face_encoding,
    val_face_encoding,
    skerritt_face_encoding
]
known_face_names = [
    "Tom Cruise",
    "Anthony Edwards",
    "Val Kilmer",
    "Tom Skerritt"
]
# Initialize some variables

# face_locations = []
# face_encodings = []
# face_names = []
process_this_frame = True


import json
import requests
import os

# host = "https://search-vsa-gaqkax4oalmlts7oxliznyrzuu.us-east-2.es.amazonaws.com"
# username = "admin"
# password = "Teampass1!"
# auth = (username, password)

# def add_request(json_name):
#     host = "https://search-vsa-gaqkax4oalmlts7oxliznyrzuu.us-east-2.es.amazonaws.com/"
#     auth = ("admin", "Teampass1!")
#     headers = {'Content-Type': 'application/json'}

#     with open(f'{json_name}.json', 'rb') as f:
#         data = f.read()
#     response = requests.post(host+"_bulk", headers=headers, data=data, auth=auth)
#     print(response.text)


def make_json(model, video_name, id, fps, descriptions, scores):
    with open("result.json", "a") as f:
        index = {
            "index": {
                "_index": video_name,
                "_type": "frame",
                "_id": id
            }
        }
        
        data = {
                "fps": fps,
                "label": [
                ]
            }

        for (description, score) in zip(descriptions, scores):
            labels = {
                        "model": model,
                        "description": description,
                        "score": score
                     }
            data["label"].append(labels)
        json.dump(index, f)
        f.write("\n")
        json.dump(data, f)
        f.write("\n")

frame_num = 0

while True:
    print(frame_num)
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # print('\rTracking frame: {}/{}'.format(frame_num + 1, len(frame)), end='')

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        print("PROCESS_THIS_FRAME")
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_distances_list = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_distances_list.append(np.min(face_distances))
            else:
                face_distances_list.append(0)

            face_names.append(name)


    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # filepath = json_filepath+'face_recog_{}_frame_{}'.format(video_name, frame_num+1)+'.json'
    # with open(filepath, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent="\t")

    # Example metadata
    video_name = 'vid_0'
    model = 'face'
    fps = 30
  
    make_json(model, video_name, frame_num, fps, face_names, face_distances_list)
    # add_request("result")

    # Display the resulting image
    cv2.imshow('Video', frame)
    frame_num += 1
    print("Successful")
    out.write(frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

