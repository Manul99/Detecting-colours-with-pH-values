import cv2
import numpy as np
from collections import deque

# Function to return the ph level based on the color
def get_ph_value(hsv_color):
    ph_ranges ={
        "pH 1 - Medium Pink": [(330, 50, 90), (345, 70, 100)],
        "pH 3 - Bronzed Flesh":[(20,60,90),(35,75,100)],
        "pH 5 - Fresh Pineapple": [(45, 60, 90), (55, 75, 100)],
        "pH 6 - Aztec Gold": [(35, 50, 70), (40, 65, 80)],
        "pH 7 - Gimblet": [(50, 40, 70), (55, 60, 75)],
        "pH 8 - Sulfuric Yellow": [(55, 40, 60), (65, 50, 65)],
        "pH 9 - Dead Flesh": [(80, 30, 60), (90, 40, 65)],
        "pH 10 - Purslane": [(85, 30, 60), (95, 35, 65)],
        "pH 12 - Green Weed": [(150, 30, 50), (160, 45, 60)],
        "pH 14 - Armada": [(170, 20, 40), (180, 25, 45)]
    }

    for ph,(lower,upper) in ph_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)

        mask = cv2.inRange(hsv_color,lower,upper)
        if cv2.countNonZero(mask) > 0:
            return ph
    return None

def detectColorInLargestContour(hsv_frame, contours):
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1,255, thickness=cv2.FILLED)
        mean_hsv = cv2.mean(hsv_frame, mask=mask)[:3]
        mean_hsv_array = np.uint8([[mean_hsv]])
        return get_ph_value(mean_hsv_array)

    return None

# Stabilization parameters
BUFFER_SIZE = 10
ph_history = deque(maxlen=BUFFER_SIZE)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640,480))

    #Convert BGR frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_threshold = np.array([0, 100, 100])
    upper_threshold = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_threshold, upper_threshold)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the detected pH value
    detected_ph = get_ph_value(hsv_frame)
    #detected_ph = detectColorInLargestContour(hsv_frame, contours)

    # Display the detected pH value if any
    if detected_ph:
        ph_history.append(detected_ph)

    if len(ph_history) > 0:
        most_common_ph = max(set(ph_history), key=ph_history.count)

        cv2.putText(frame, most_common_ph, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



    #Display the frame
    cv2.imshow('pH Color Detection',frame)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()

