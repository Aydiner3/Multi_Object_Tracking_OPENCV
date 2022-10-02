import string
import cv2

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}
'''
Opencv üzerinde çalışan yapay zeka algoritmalarıdır. Bunlardan en güçsüz olanı
boosting algoritmasıdır
Bazı versiyonlarda cv2.legacy.... şeklinde çağrılabilir.
'''

tracker_name = "medianflow" # medianflow algoritması için

trackers = cv2.legacy.MultiTracker_create() # birden çok nesneyi tanımak için verilen
# class içindeki fonksiyondur

video_path = "MOT17-04-DPM.mp4"

cap = cv2.VideoCapture(video_path)

fps = 30 # frame per second olması gereken default değer 30' dur. Ve saniyede
# gelen frame sayısını gösterir.
f = 0

while True:
    ret, frame = cap.read()
    (H, W) = frame.shape[:2] # Bu şekilde shape'in ilk iki indexini alarak
    # yükseklik ve genişlik değerlerini alırız.

    frame = cv2.resize(frame, dsize=(960,480)) # frame yeniden boyutlandırılır.

    (success, boxes) = trackers.update(frame) # tracker' ı update eder her frame
    # için; kutucukları ve success değerlerini döndürür.

    info = [
        ("Tracker", tracker_name),
        ("Success", "Yes" if success else "No")
    ] # tracker ve Success bilgi atamaları yapılır.

    string_text = ""

    for (i, (k,v)) in enumerate(info): # key ve value değerleri enumarate edilir.
        text = "{}: {}".format(k,v)
        string_text = string_text + text + " "
    
    cv2.putText(frame, string_text, (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255), 2)

    for box in boxes: # kutucukları alma işlemidir yani rects' leri...
        (x,y,w,h) = [int(v) for v in box]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("t"): # Boşluk işlemidir.
        box = cv2.selectROI("Frame", frame, fromCenter=False) # region of interest
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        trackers.add(tracker, frame, box)
        '''
        Kutucukları tracker içine veririz ki  takip edebilelim. Birden fazla tracker
        olduğu için birden fazla objeyi takip edebiliyoruz. MultiTracker_create sayesinde
        '''
    elif key == ord("q"): break

    f = f + 1 # Sadece herhangi bir zamanda frame göstermek için assign edildi.

cap.release()
cv2.destroyAllWindows()


