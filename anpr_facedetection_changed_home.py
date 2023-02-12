import cv2
import numpy as np
import imutils
import easyocr
import datetime
import csv
import re
import tkinter as tk
window = tk.Tk()
current_time = datetime.datetime.now()
time = current_time.strftime("%Y-%m-%d %H:%M:%S")

cap = cv2.VideoCapture('C:/Users/PC/PycharmProjects/ANPR/Dataset/video2.mp4')
#extra addition
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

license_plate_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #new line added
    # faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = frame[y:y + h, x:x + w]
    #     cv2.imwrite('face.jpg', roi_color)
    #     print("face saved")

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 50, 150)  # Adjust Canny edge detection threshold

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Adjust contour area threshold
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)  # Adjust approximation accuracy
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        continue

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(frame, frame, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = new_image[x1:x2 + 1, y1:y2 + 1]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    print(result)
    print(type(result))
    if len(result)==0:
        print("cannot read numberplate ")
    if len(result[0][-2])<4:
        text=result[1][-2]
    else:
        text = result[0][-2]

    text = text.replace(" ", "")
    if text:
        #new added
        table = str.maketrans({symbol: None for symbol in "!@#$%^&*()_+-=[]{};':\"<>,.?/\\|"})
        #new added
        cleaned_text = text.translate(table)
        license_plate_detected = True
        #cv2.imwrite('detected_plate.jpg', cropped_image)
        print(result)
        print(cleaned_text)


        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = 10
        text_y = approx[2][0][1] + 60
        res = cv2.putText(frame, text=cleaned_text, org=(text_x, text_y), fontFace=font, fontScale=2,
                          color=(255, 0, 0), thickness=7, lineType=cv2.LINE_AA)
        res = cv2.rectangle(frame, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 5)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        resized_result = cv2.resize(res, (700, 550))
        #cv2.imshow("result", resized_result)

        with open('C:/Users/PC/PycharmProjects/ANPR/vehicles.csv','w') as f:
            f.write(f'\n{cleaned_text},{time}')
        print('License plate detected and saved!')

        with open('numbers.csv','r') as f:
            csvreader = csv.reader(f)
            print("csv reader is",csvreader)
            number_plates = [row[0] for row in csvreader]
            print("numberplate is ",number_plates)


        # Check if the number plate extracted from the video exists in the list of number plates
        if cleaned_text in number_plates:
            message=f"{cleaned_text} is a registered number plate"
            label = tk.Label(text=message,font=("Arial", 16, "bold"))
            label.pack()
            window.mainloop()
        else:
            msg=f"{cleaned_text} is not a registered number plate,you can register the vehicle"
            label = tk.Label(text=msg,font=("Arial", 16, "bold"))
            label.pack()
            window.mainloop()
        break
cv2.waitKey(0)
