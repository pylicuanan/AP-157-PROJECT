## automatic 10 secs
import cv2 as cv
import pandas as pd
import numpy as np
import datetime
import time
import TemplateMatching

tails_file = 'Tails.jpg'
heads_file = 'Heads.jpg'
flag = cv.IMREAD_COLOR_BGR
method = cv.TM_CCOEFF_NORMED
tails_threshold = 0.4
heads_threshold = 0.6

# Initialize camera
cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_EXPOSURE,-4)


# Set video codec and output settings
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = None
recording = False
auto_capture = False
last_capture_time = 0
image_counter = 0  # Counter for number of auto-captured images
capture_interval = 10  # seconds

num_heads = 0
num_tails = 0

print(f"Press 'c' to start automatic image capture every {capture_interval} seconds")
print("Press 'r' to reset counter")
print("Press 'q' to quit")

data = {
    "capture_count": image_counter,
    "num_heads": num_heads,
    "num_tails": num_tails
}
df = pd.DataFrame([data])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    current_time = time.time()

    # Time remaining for next capture
    if auto_capture:
        time_remaining = max(0, int(capture_interval - (current_time - last_capture_time)))
    else:
        time_remaining = 0

    # Automatically capture image every 10 seconds
    if auto_capture and (current_time - last_capture_time) >= capture_interval:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"captures/capture_{timestamp}.jpg"
        cv.imwrite(img_filename, frame)
        image_counter += 1
        print(f"[Auto] Image saved: {img_filename}")
        last_capture_time = current_time
        
        image = cv.imread(img_filename, cv.IMREAD_UNCHANGED)
        if image.size:
            result_file = f"results/result_{timestamp}.jpg"
            result = TemplateMatching.coin_match(img_filename, heads_file, tails_file, result_file,
                                                         flag, method, 
                                                         heads_threshold, tails_threshold)
            num_heads += result.num_heads
            num_tails += result.num_tails
            
            if result.num_heads!=0 or result.num_tails!=0:
                new_data = {
                    "capture_count": image_counter,
                    "num_heads": result.num_heads,
                    "num_tails": result.num_tails
                }
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            
        else:
            print('No image')


    # Overlay image counter
    overlay_text = f"Images Captured: {image_counter}"
    cv.putText(frame, overlay_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, f"Heads: {num_heads}", (480, 30), cv.FONT_HERSHEY_SIMPLEX,
                1, (255,255,0), 2, cv.LINE_AA)
    cv.putText(frame, f"Tails: {num_tails}", (480, 70), cv.FONT_HERSHEY_SIMPLEX,
                1, (255,255,0), 2, cv.LINE_AA)

    # Overlay countdown timer
    if auto_capture:
        timer_text = f"Next capture in: {time_remaining}s"
        cv.putText(frame, timer_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv.LINE_AA)

    # Show video feed
    cv.imshow('Live Camera', frame)

    # Wait for key press
    key = cv.waitKey(1) & 0xFF

    if key == ord('c'):
        if not auto_capture:
            auto_capture = True
            last_capture_time = time.time()
            print("Automatic capture every 5 seconds started.")
        else:
            auto_capture = False
            print("Automatic capture stopped.")
            
            if df.shape[0]>1:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                data_filename = f"datasets/data_{timestamp}.csv"
                df.to_csv(data_filename, index=False)

    elif key == ord('r'):
        num_heads = 0
        num_tails = 0
        print("Coin face counter reset.")
        if auto_capture:
            auto_capture = False
            print("Automatic capture stopped.")
        
        if df.shape[0]>1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"datasets/data_{timestamp}.csv"
            df.to_csv(data_filename, index=False)
            

    elif key == ord('q'):
        print("Quitting...")
        break

# Release resources
cap.release()
if out:
    out.release()
cv.destroyAllWindows()