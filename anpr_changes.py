import cv2
import numpy as np
import imutils
import easyocr

cap = cv2.VideoCapture('C:/Users/DELL/PycharmProjects/number_plate/video2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    text = result[0][-2] if result else ''
    print(text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #res = cv2.putText(frame, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                      #color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
    res = cv2.putText(frame, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=2,
                      color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)

    res = cv2.rectangle(frame, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 5)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    res = cv2.rectangle(frame, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 5)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    resized_result = cv2.resize(res, (800, 450))
    cv2.imshow("result", resized_result)
    #cv2.imshow("result", res)
    cv2.waitKey(0)

    with open('vehicles.csv', 'w') as f:
        f.write(f'\n{text}')

