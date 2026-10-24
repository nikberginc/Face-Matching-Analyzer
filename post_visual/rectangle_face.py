import cv2, os

def face_rac(image_path, image_locaton, age, gender):

    #program will read the image by cv2 function
    image = cv2.imread(image_path)

    #the locations of the photos will be unboxed into more veriables
    top, right, bottom, left = image_locaton

    #cv2 will drew rectangle where the face locations are
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)




    #program will add the age group number


    #the settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)
    thickness = 2
    text_position = (left, top - 15)
    label = f"{gender}, {age}"

    #cv2 will draw the number group on the picture
    cv2.putText(image, label, text_position, font, font_scale, color, thickness)





    #cv2 will store the image in this path
    output_path = os.path.join(os.environ["USERPROFILE"], "Documents", "FaceComperesment", "matches", os.path.basename(image_path))
    cv2.imwrite(output_path, image)
   