import face_recognition

def vectorCREATE(image_format, face_locations):

    face_encodings = face_recognition.face_encodings(image_format, face_locations)
    return face_encodings