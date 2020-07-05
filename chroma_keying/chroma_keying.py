# chroma_keying

import cv2
import numpy as np

WINDOW_NAME = 'Just Click on background'

PICKED_COLOR = None
FRAME = None
FRAME_LIST = []

def get_mask(im):
    global PICKED_COLOR

    if PICKED_COLOR is None:
        return None

    picked_color = np.array([[PICKED_COLOR]], dtype=np.uint8)
    
    hsv_color = cv2.cvtColor(picked_color, cv2.COLOR_BGR2HSV)

    ph, ps, pv = hsv_color[0][0]

    h = cv2.getTrackbarPos('Tolerance H', WINDOW_NAME)
    lh = max(0, ph - h)
    hh = min(179, ph + h)

    s = cv2.getTrackbarPos('Tolerance S', WINDOW_NAME)
    ls = max(0, ps - s)
    hs = min(255, ps + s)

    v = cv2.getTrackbarPos('Tolerance V', WINDOW_NAME)
    lv = max(0, pv - v)
    hv = min(255, pv + v)

    lower = np.array([lh, ls, lv])
    upper = np.array([hh, hs, hv])

    img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, lower, upper)

    alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    blured_alpha = cv2.GaussianBlur(alpha, (5, 5), 0, 0)
    alpha = cv2.cvtColor(blured_alpha, cv2.COLOR_BGR2GRAY)

    t = cv2.getTrackbarPos('T', WINDOW_NAME)
    _, thrsh = cv2.threshold(alpha, 255-t, 255, cv2.THRESH_BINARY)
    
    thrsh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    thrsh = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    return thrsh

def process(im, bg):
   
    mask = get_mask(im)
    if mask is None:
        return im

    masked = cv2.bitwise_and(im, im, mask=cv2.bitwise_not(mask))
    
    bg = cv2.resize(bg, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
    bg = cv2.bitwise_and(bg, bg, mask=mask)

    final = bg + masked

    return final

def mouse_cb(event, x, y, flags, param):
    global PICKED_COLOR, FRAME_LIST

    if FRAME is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        frame_array = np.array(FRAME_LIST)
        m = frame_array.mean(axis=0)

        if  m.shape[1] + 10 < x < m.shape[1] - 10 and m.shape[0] + 10 < y < m.shape[0] - 10:
            PICKED_COLOR = m[y-10:y+10,x-10:x+10].mean(axis=0).mean(axis=0).astype(np.uint8)
        else:
            PICKED_COLOR = m[y][x].astype(np.uint8)

def main():
    global FRAME

    # webcam
    cap = cv2.VideoCapture(0)

    # cap = cv2.VideoCapture('greenscreen-asteroid.mp4')
    _, FRAME = cap.read()
    
    bg = cv2.imread('./data/bg.png')

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    def dummy(*args, **kwargs):
        pass
    
    cv2.createTrackbar('Tolerance H', WINDOW_NAME, 30, 179, dummy)
    cv2.createTrackbar('Tolerance S', WINDOW_NAME, 140, 255, dummy)
    cv2.createTrackbar('Tolerance V', WINDOW_NAME, 160, 255, dummy)
    cv2.createTrackbar('T', WINDOW_NAME, 55, 254, dummy)
    cv2.createTrackbar('on/off', WINDOW_NAME, 0, 1, dummy)

    while True:

        _, FRAME = cap.read()
        if FRAME is None:
            break

        k = cv2.waitKey(30)
        if k == 27: # ESC
            break

        if len(FRAME_LIST) < 17:
            FRAME_LIST.append(FRAME)
        else:
            FRAME_LIST.pop(0)
            FRAME_LIST.append(FRAME)


        img = process(FRAME, bg)
        cv2.imshow(WINDOW_NAME, img if cv2.getTrackbarPos('on/off', WINDOW_NAME) == 0 else FRAME)

        

    cap.release()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()

