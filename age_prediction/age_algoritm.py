import cv2

def age_pre(img_face):

    #models were trained on 8 diffrent groups, thats why we use 8 diffrent groups
    AGE_LIST = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
    ]   



    #the program willload the pre-trained model
    prototxt_path = "models/age_deploy.prototxt"
    model_path = "models/age_net.caffemodel"

    age_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


    #we will put the face of the image to a standerdized format(size, coler, reshaped...)
    blob = cv2.dnn.blobFromImage(img_face, scalefactor=1.0, size=(255, 255), mean=(78.426, 87.768, 114.896), swapRB=False)

    #program feeds the pre-traied models the normalized image "blob"
    age_net.setInput(blob)




    #pre-traied model throus the list of the ages and how much confident is for each group --> the .argmax will cosee the biggest confidence and give the posision --> the number
    #will be sent to the AGE_LIST[number] to give the specific group in the list
    age = AGE_LIST[age_net.forward()[0].argmax()]

    return age

