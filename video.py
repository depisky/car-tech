import cv2
import threading


# def stop_pls():
#     print("STOP")


# def unpause_video():
paused = False




# paused = False

def play_vid(paused):
    cap = cv2.VideoCapture('C:\LAB\car tech\carvid.mp4')
    while True:
        ret, frame =cap.read()
        if ret:
            if paused==False:
                cv2.imshow('Video', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    cap.release()
                    break
            
