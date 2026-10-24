import face_recognition, os
from face_processing.formatinImage import formating
from face_processing.FaceLocation import faceLOCATE
from face_processing.VectorCreate import vectorCREATE
from mapping.Newfile_creation import NEWfile
from post_visual.rectangle_face import face_rac
from age_prediction.age_algoritm import age_pre
from gender_prediction.gender_algoritm import gender_pre

def faceALGORITM(path_list, org_image_vector):

    check_25 = False
    check_50 = False
    check_75 = False



    image_detected_list_path = []
    gender_list = []
    age_list = []
    locations = []




    print("\n\n\n\nStarting the process...")

    #program will go throu every photo in the folder
    for i in range(len(path_list)):

        #for visuals!!! when the program will be 25% or 50% or 75% done it will display it
        if   i >= (1/4) * len(path_list) and not check_25:
            check_25 = True
            print("\n\n\nProgress: 25% of images processed")
        
        elif i >= (2/4) * len(path_list) and not check_50:
            check_50 = True
            print("Progress: 50% of images processed")
        
        elif i >= (3/4) * len(path_list) and not check_75:
            check_75 = True
            print("Progress: 75% of images processed")


      
        

        #program calls the functions that gets the image in a format so the face_recognision works
        image_format = formating(path_list[i])

        #program calls a function to get the face location in a image
        image_location = faceLOCATE(image_format)


        #program will call the function to get the vector of the image face. IF THERE ARE MORE FACES, THE IMAGE_VECTOR
        image_vector = vectorCREATE(image_format, image_location)


        #program will use every vector of faces if there are any to compare them each one to the original image until the similar face is detected
        for x in range(len(image_vector)):
            #                                                                           the [0] is there so after the function ands you dont get [0.45] but 0.45
            distance = face_recognition.face_distance([org_image_vector], image_vector[x])[0]

            #if distance is less than 0.6 it will store this image into the new folder - - FaceComperesment --> matches
            if distance < 0.6:

                #program calls the function to create the folder - - FaceComperesment --> matches
                NEWfile("matches")


                #program will store the image paths of the imiges with the same face
                image_detected_list_path.append(path_list[i])


                #the locations of the photos will be unboxed into more veriables
                top, right, bottom, left = image_location[x]

                #this crops the face out of the image
                imag_face = image_format[top:bottom, left:right]

                #calls the funtion that gives a age predicted gorup of maching face on the image and is stored in a list
                age_list.append(age_pre(imag_face))

                #calls the funtion that gives a gender predicted gorup of maching face on the image
                gender_list.append(gender_pre(imag_face))

                #will store the face locations in a list
                locations.append(image_location[x])

            


    male_count = 0
    female_count = 0

    #program will get the avrage gender in images
    for g in gender_list:
        if g == "Male":
                    male_count += 1
        elif g == "Female":
            female_count += 1

    if male_count>female_count:
        gender = "Male"

    elif male_count<female_count:
        gender = "Female"
    else:
        gender = "Undefined"




    #program will go throu every photo with the same face again
    for i in range(len(image_detected_list_path)):
         
        #calls the function that draws a ractangle where the face is and adds gender, age.
        face_rac(image_detected_list_path[i], locations[i], age_list[i], gender)


