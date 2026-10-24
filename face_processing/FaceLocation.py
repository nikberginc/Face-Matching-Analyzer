import face_recognition

def faceLOCATE(formated_image):

    #program gets the cordinats where the face is on the image
    face_location = face_recognition.face_locations(formated_image)
    return face_location