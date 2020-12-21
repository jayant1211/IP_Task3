import cv2
import pytesseract
import os
from gtts import gTTS   
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


for filename in os.listdir("Text_Images"):
    #print(filename)
    img = cv2.imread("Text_Images/" + filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Performing OTSU threshold 
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    #cv2.imshow("thresh", cv2.resize(thresh1, (1000, 700)))
    # Appplying dilation on the threshold image 
    dilation = cv2.dilate(thresh1, (3, 3), iterations = 1)
    #cv2.imshow("dilation", cv2.resize(dilation, (1000, 700))) 
    


    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh1 > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the img to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output img
    # print("[INFO] angle: {:.3f}".format(angle))
    
    cv2.imshow("Input", img)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)


    # # Finding contours 
    # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
    #                                                 cv2.CHAIN_APPROX_NONE) 
    
    # Looping through the identified contours 
    # Then rectangular part is cropped and passed on 
    # to pytesseract for extracting text from it 
    # Extracted text is then written into the text file 
    # for cnt in contours: 
    #     x, y, w, h = cv2.boundingRect(cnt) 
        
    #     # Drawing a rectangle on copied image 
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2) 

    text = pytesseract.image_to_string(rotated)
    #print(type(text))

    file = open("Text_Recognized/" + os.path.splitext(filename)[0] + ".txt", "w+") 
    file.write(text) 
    file.close() 
    print("\nText file saved")

    # Language in which you want to convert 
    language = 'en'
    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    myobj = gTTS(text=text, lang=language, slow=False) 
    
    # Saving the converted audio in a mp3 file named 
    # welcome  
    try:
        myobj.save("Text_Audio/" + os.path.splitext(filename)[0] + ".mp3")  
        print("Audio file saved\n")
    except:
        print("Unexpected error, audio unsaved :(\n")
    

for filename in os.listdir("Text_Audio"):
    #print(filename)
    # Playing the converted file 
    audioPath = "Text_Audio/" + filename
    #print(audioPath)
    os.system("start " + audioPath)
    print("Executing audio file...\n")