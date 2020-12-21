import cv2
import pytesseract
import os
from gtts import gTTS   

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

    text = pytesseract.image_to_string(dilation)
    #print(type(text))

    file = open("Text_Recognized/" + filename + ".txt", "w+") 
    file.write(text) 
    file.close() 
    print("Text file saved")

    # Language in which you want to convert 
    language = 'en'
    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    myobj = gTTS(text=text, lang=language, slow=False) 
    
    # Saving the converted audio in a mp3 file named 
    # welcome  
    myobj.save("Text_Audio/" + os.path.splitext(filename)[0] + ".mp3")  
    print("Audio file saved")

    #cv2.imshow("img", cv2.resize(img, (1000 ,700)))
    #cv2.waitKey(0)

for filename in os.listdir("Text_Audio"):
    #print(filename)

    # Playing the converted file 
    audioPath = "Text_Audio/" + filename
    #print(audioPath)
    os.system("start " + audioPath)
    print("Executing audio file")