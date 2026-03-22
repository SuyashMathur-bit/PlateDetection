import os

import cv2
import numpy as np
import pytesseract
import re

path = "Plate_1711623405530_1711623405684.png"
vehicle_db = {}
txt_path="database.txt"
if os.path.exists(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 4:

                plate_key = parts[0].strip().upper().replace(" ", "")
                vehicle_db[plate_key] = {
                    "OwnerName": parts[1].strip(),
                    "VehicleModel": parts[2].strip(),
                    "status": parts[3].strip(),
                    "fine": parts[4].strip() if len(parts) > 4 else "0"
                }
    print(f"Database Loaded: {len(vehicle_db)} records.")
else:
    print("Error: database.txt not found!")

plateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
img = cv2.imread(path)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
number_plates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

for (x, y, w, h) in number_plates:
    area = w * h
    if area > 500:
        imgRoi = img[y:y + h, x:x + w]


        imgRoi = cv2.resize(imgRoi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        raw_text = pytesseract.image_to_string(thresh, config=custom_config)


        plate_number = re.sub(r'[^A-Z0-9]', "", raw_text.upper())



        if plate_number.strip():
            print(f"Successfully Detected: {plate_number}")

        detected_plate = plate_number.strip().upper().replace(" ", "")


        if detected_plate in vehicle_db:

            person = vehicle_db[detected_plate]

            print("\n" + "=" * 40)
            print(f"      MATCH FOUND: {detected_plate}")
            print("=" * 40)
            print(f"Owner:   {person['OwnerName']}")
            print(f"Vehicle: {person['VehicleModel']}")
            print(f"Status:  {person['status']}")
            print(f"Fine:    RS.{person['fine']}")
            print("=" * 40 + "\n")

        else:
            # This only runs if the car is NOT in your notepad file
            print(f"Detected Plate [{detected_plate}] is not registered in the database.")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("OCR Input (Thresh)", thresh)

# Final display
cv2.imshow("Final Result", img)

print("\nProcessing complete. Press any key on the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()