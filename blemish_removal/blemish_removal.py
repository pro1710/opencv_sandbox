import cv2
import numpy as np

DEBUG = False

orig  = cv2.imread('./data/blemish.png', cv2.IMREAD_COLOR)
img = orig.copy()

HEIGHT, WIDTH = img.shape[0], img.shape[1]
WINDOW_NAME = 'BlemishRemoval'
RADIUS = 16

# tmp
if DEBUG:
    blemish = 'blemish'
    patch = 'patch'


def validate(x, y):
    if WIDTH - 2*RADIUS < x or x < 0:
        return False 

    if HEIGHT - 2*RADIUS < y or y < 0:
        return False 

    return True

def get_top_left(center):
    x, y = center
    tx = min(max(0, x-RADIUS), WIDTH-2*RADIUS) 
    ty = min(max(0, y-RADIUS), HEIGHT-2*RADIUS)

    return tx, ty
    
def evaluate(top_left):
    global img
    x, y = top_left

    im = cv2.cvtColor(orig[y:y+2*RADIUS, x:x+2*RADIUS], cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(im, cv2.CV_8U, ksize = 3,
                             scale = 1, delta = 0)

    _, lap = cv2.threshold(lap, 120, 255, cv2.THRESH_BINARY)

    val = lap.mean()
    
    if DEBUG:
        cv2.rectangle(img, (x, y), (x+2*RADIUS, y+2*RADIUS), (0, 0, 255), 1)
        img = cv2.putText(img, str(round(val)), (x, y+RADIUS), cv2.FONT_HERSHEY_SIMPLEX,  0.2, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('sobel', lap)

    return val

def find_patch(top_left):
    cx, cy = top_left

    lookAroudList = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    patch = None
    min_val = None

    for kx, ky in lookAroudList:
        tx, ty = cx + kx*(2*RADIUS), cy + ky*(2*RADIUS)
        if (validate(tx, ty)):

            val = evaluate(top_left=(tx, ty))
            if min_val == None or min_val > val:
                min_val = val
                patch = (tx, ty)

    return patch

def mouse_on_click(event, x, y, flags, param):
    global img, orig
    if event == cv2.EVENT_LBUTTONDOWN:
        img = orig.copy()

    elif event == cv2.EVENT_LBUTTONUP:

        
        tx, ty = get_top_left(center=(x, y))

        px, py = find_patch(top_left=(tx, ty))

        pimg = orig[py:py+2*RADIUS, px:px+2*RADIUS]

        if DEBUG:
            cv2.circle(img, (x, y), RADIUS, (0, 255, 0), 1)
            bimg = img[ty:ty+2*RADIUS, tx:tx+2*RADIUS]
            
            cv2.imshow(blemish, bimg)
            cv2.imshow(patch, pimg)

        src_mask = np.zeros((pimg.shape[0], pimg.shape[1]), np.uint8) 
        cv2.circle(src_mask, (pimg.shape[0]//2, pimg.shape[1]//2), RADIUS, 255, thickness=-1)

        orig = cv2.seamlessClone(pimg, orig, src_mask, (tx+RADIUS, ty+RADIUS), cv2.NORMAL_CLONE)

    if DEBUG:
        cv2.imshow(WINDOW_NAME, img)
    else:
        cv2.imshow(WINDOW_NAME, orig)


def main():
   
    if img is None:
        print('Error: image file not found')
        return

    cv2.imshow(WINDOW_NAME, orig)

    cv2.setMouseCallback(WINDOW_NAME, mouse_on_click)

    ch = cv2.waitKey()


if __name__=='__main__':
    main()
    cv2.destroyAllWindows()
