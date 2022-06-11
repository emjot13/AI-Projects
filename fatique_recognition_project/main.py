
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import threading
from playsound import playsound
from pynput import keyboard
import argparse

def eye_aspect_ratio(eye):                                                        # calculates openness of an eye
    A = dist.euclidean(eye[1], eye[5])                          
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def final_ear(shape):
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]       # detects the eyes and applies above function
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    left_eye = shape[left_eye_start:left_eye_end]
    right_eye = shape[right_eye_start:right_eye_end]
    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)
    ear = (left_EAR + right_EAR) / 2.0
    return (ear, left_eye, right_eye)


def lip_distance(shape):                                            # calculates openness of a mouth
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def camera(frame, eye, gray, rect, predictor, ear):                # enables real-time view from camera
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    distance = lip_distance(shape)
    left_eye = eye[1]
    right_eye = eye[2]
    left_eye_convex_hull = cv2.convexHull(left_eye)
    right_eye_convex_hull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [left_eye_convex_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [right_eye_convex_hull], -1, (0, 255, 0), 1)
    lip = shape[48:60]
    cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.imshow("Fatique detector", frame)

def alarm(filepath):                    # plays sound alarm
    playsound(filepath)

# global vs, multiple_threads, new_thread 


def on_press(key, abortKey='esc'):    
    try:
        k = key.char 
    except:
        k = key.name      
    if k == abortKey:
        # if multiple_threads:
        #     new_thread.join()     # was just testing
        # vs.stop()
        return False  



def get_args():                                                         # gets arguments from command line flags
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cycles", help="number of cycles",type=int, default= 3)
    parser.add_argument("-s", "--seconds", help="seconds for one cycle", type=int, default=30)
    parser.add_argument("-v", "--verbose", help="Verbose output", type=bool, default=False)
    parser.add_argument("-a", "--audio", help="Audio output", type=bool, default=False)
    parser.add_argument("-cetl" or "--closed-eyes-time-limit", "--closed_eyes_seconds_treshold", help="Time after which closed eyes are counted as sleeping", type=int, default=7)
    parser.add_argument("-f", "--FPS", help="Frames per second", type=int, default=24)
    parser.add_argument("-e", "--e_a_r", help="eye aspect ratio", type=float, default=0.27)
    parser.add_argument("-y", "--yawn_treshold", help="yawn treshold", type=float, default=20.0)
    parser.add_argument("-t", "--time-limit", help="if time limit is to be applied", type=bool, default=True)
    parser.add_argument("-file", '--filepath', help="path to the file to write to")
    args = parser.parse_args()
    return args.__dict__



def fatique_detector(seconds, verbose, audio, closed_eyes_seconds_treshold, FPS, e_a_r, yawn_treshold, time_limit):  # main detecting function
    # global vs, multiple_threads, new_thread
    default_thread_number = threading.active_count()
    closed_eyes_treshold_fps = closed_eyes_seconds_treshold * FPS 
    sleeping_counter, yawning_counter, frame_counter = 0, 0, 0
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    vs = VideoStream(framerate=FPS).start()                        # starts video
    time.sleep(1)
    yawn, sleep, multiple_threads = False, False, False
    starting_time = time.time()
    timeout = starting_time + seconds
    while True:        
        frame = imutils.resize(vs.read(), width=450)
        to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(to_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = face_utils.shape_to_np(predictor(to_gray, rect))
            eye = final_ear(shape)
            ear = eye[0]
            distance = lip_distance(shape)
            if sleep and ear > e_a_r:
                sleeping_counter += 1
                frame_counter = 0
                sleep = False
            elif ear < e_a_r:
                frame_counter += 1
                if frame_counter >= closed_eyes_treshold_fps:
                    if audio and threading.active_count() < default_thread_number + 2:
                        multiple_threads = True
                        new_thread = Thread(target=playsound, args=('wake_up.aac', ))
                        new_thread.start()
                    cv2.putText(frame, "WAKE UP!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    sleep = True
            else:
                frame_counter = 0
            if distance > yawn_treshold:
                cv2.putText(frame, "FATIQUE ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawn = True
            elif yawn and distance < yawn_treshold:
                yawn = False
                yawning_counter += 1
            if verbose:
                camera(frame, eye, to_gray, rect, predictor, ear)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('z'):
            if multiple_threads:
                new_thread.join()
            cv2.destroyAllWindows()
            vs.stop()
            exit(0)
        if key == ord("q") or (time_limit and time.time() > timeout):
            if multiple_threads:
                new_thread.join()
            cv2.destroyAllWindows()
            vs.stop()
            print(f"In the last {round(time.time() - starting_time)} seconds you've yawned: {yawning_counter} times and had your eyes closed for longer than {closed_eyes_seconds_treshold} seconds {sleeping_counter} times")
            return sleeping_counter, yawning_counter



def main(args):
    for _ in range(args['cycles']):
        fatique_detector(args['seconds'], args['verbose'], args['audio'], args['closed_eyes_seconds_treshold'], args['FPS'] , args['e_a_r'], args['yawn_treshold'], args['time_limit'])


        
if __name__ == '__main__':
    args = get_args()
    if not args['verbose']:
        listener = keyboard.Listener(on_press=lambda event: on_press(event))
        listener.start()  
        Thread(target=main, args=(args, ), name='main', daemon=True).start()
        listener.join() 
    else:
        main(args)
