from ultralytics import YOLO
import cv2
import math 
from video import play_vid
from video import paused
import time 
import threading
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def the_main_loop(cap, model, classNames):
    internal_value_for_first_set_up = 1

    while True:
        # Read a frame from the video capture
        success, img = cap.read()
        if not success:
            print("Error: Failed to read frame from video capture")
            break

        # Perform object detection using the model
        results = model(img, stream=True)

        # Process detection results
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Extract confidence and class
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Display class name
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                # Check x-coordinate value
                print(org)
                x_value = org[0]
                if ((x_value>100) and (x_value<300)):
                    print("Object in the road - Stop car: " +str(x_value)) 
                    paused=True
                    print(paused)
                    # stop_my_pain = threading.Thread(target= stop_pls ,daemon=False).start()
                else:
                    paused=False
                    print(paused) 
                    # unpause = threading.Thread(target= unpause_video ,daemon=False).start()
 
                if internal_value_for_first_set_up == 1:
                    vidplay = threading.Thread(target=play_vid, args=(paused), daemon=False).start()


                    internal_value_for_first_set_up = internal_value_for_first_set_up+1

        # Display processed frame
        cv2.imshow('Webcam', img)

        # Check for user input to quit
        if cv2.waitKey(1) == ord('q'):
            break

        # Pause execution for the specified interval
        

    # Release video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
human_decetion_thread = threading.Thread(target=the_main_loop, args=(cap, model, classNames,), daemon=False).start()





# internal_value_for_first_set_up=1
# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)

#     # coordinates
#     for r in results:
#         boxes = r.boxes

#         for box in boxes:
#             # bounding box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

#             # put box in cam
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

#             # confidence
#             confidence = math.ceil((box.conf[0]*100))/100
#            # print("Confidence --->",confidence)
#             # class name
#             cls = int(box.cls[0])
#             #print(box.xyxy)
#             #print("Class name -->", classNames[cls])
            
#             # object details
#             org = [x1, y1]
#             x_value=org[0]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             color = (255, 0, 0)
#             thickness = 2
#             if ((x_value>100) and (x_value<250)):
#                 print("object in road stop car"+classNames[cls])
#                 threading.Thread(target=stop_video).start
#             else:
#                 threading.Thread(target=unpause_video).start 
#             if internal_value_for_first_set_up==1:
#                 threading.Thread(target=play_vid).start
#                 internal_value_for_first_set_up=internal_value_for_first_set_up+1

                
#             # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
#             cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
#             print (str(x_value)+"x value ")

#     cv2.imshow('Webcam', img)
#     if cv2.waitKey(1) == ord('q'):
#         break


    
                
# cap.release()
# cv2.destroyAllWindows()










# import numpy as np
# import cv2
 
# # initialize the HOG descriptor/person detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# cv2.startWindowThread()

# # open webcam video stream
# cap = cv2.VideoCapture(0)

# # the output will be written to output.avi
# out = cv2.VideoWriter(
#     'output.avi',
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640,480))

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # resizing for faster detection
#     frame = cv2.resize(frame, (640, 480))
#     # using a greyscale picture, also for faster detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

#     # detect people in the image
#     # returns the bounding boxes for the detected objects
#     boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

#     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

#     for (xA, yA, xB, yB) in boxes:
#         # display the detected boxes in the colour picture
#         cv2.rectangle(frame, (xA, yA), (xB, yB),
#                           (0, 255, 0), 2)
    
#     # Write the output video 
#     out.write(frame.astype('uint8'))
#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# # and release the output
# out.release()
# # finally, close the window
# cv2.destroyAllWindows()
# cv2.waitKey(1)
