import time, os
from face_processing.formatinImage import formating
from face_processing.FaceLocation import faceLOCATE
from face_processing.VectorCreate import vectorCREATE
from mapping.ImageGetter import getter
from  face_algoritm.Algoritm_for_face_SImilarity import faceALGORITM

print("\n\n\n\n\nProgram starting...")
time.sleep(0.5)
print("Program running!")
time.sleep(0.5)

#if the user didnt insert the real path it will expect it and print the massage that it didnt find it
try:
    #user will have to insert a path of the image
    image_path = input("\n\nInsert the path to your image here (C:/Users/.../image.png): \n\n»» ")
    image_path = image_path.replace("\\", "/")


    with open(image_path, "rb") as f:
        pass


    #user will have to insert a path of the file to compare
    compere_check = input("\n\nInsert the path to your file of images to compare here (C:/Users/.../file): \n\n»» ")
    compere_check = compere_check.replace("\\", "/")


    if not os.path.exists(compere_check):
        raise FileNotFoundError(f"Directory not found: {compere_check}")
        


except FileNotFoundError:
    print("\n\n\nFile not found! Please check the file path!")  
    exit(1)







#program calls the functions that gets the original image in a format so the face_recognision works
org_image_format = formating(image_path)

#program calls a function to get the face location in the image
org_image_location = faceLOCATE(org_image_format)

#program will call the function to get the vector of the original image face
org_image_vector = vectorCREATE(org_image_format, org_image_location)
print(org_image_vector)

if not org_image_vector:
    raise ValueError("\n\n\nNo face found in original image")

org_image_vector = org_image_vector[0]




#the program calls the function to get every path of the images in a folder to one list
path_list = getter(compere_check)

#the program calls the algoritm to get the imiges with the same face in to the folder -- FaceComperesment --> matches
faceALGORITM(path_list, org_image_vector)
print("\n\nSystem has completed 100% of the task")