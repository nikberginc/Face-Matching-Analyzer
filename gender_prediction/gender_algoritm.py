import cv2

def gender_pre(img_face):


    #models were trained on 2 diffrent groups, thats why we use 2 diffrent groups
    GENDER_LIST = [
    "Male", "Female"
    ] 
    
    #the program willload the pre-trained model
    prototxt_path = "models/gender_deploy.prototxt"
    model_path = "models/gender_net.caffemodel"

    gender_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


    #we will put the face of the image to a standerdized format(size, coler, reshaped...)
    blob = cv2.dnn.blobFromImage(img_face, scalefactor=1.0, size=(227, 227), mean=(78.426, 87.768, 114.896), swapRB=False)

    #program feeds the pre-traied models the normalized image "blob"
    gender_net.setInput(blob)



    #pre-traied model throus the list of the genders and how much confident is for one of two groups --> the .argmax will choose the biggest confidence and give the posision --> the number
    #will be sent to the AGE_LIST[number] to give the specific group in the list
    gender = GENDER_LIST[gender_net.forward()[0].argmax()]


    




    return gender

