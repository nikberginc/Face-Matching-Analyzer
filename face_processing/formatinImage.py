import face_recognition

def formating(image_path: str):

    #program will use face_recognition comand so the photo will be stored in veriable image in a format that face_recognision can understand
    image = face_recognition.load_image_file(image_path)
    return image

    

    