import cv2
import numpy as np

DEBUG = False

def showImgInNewWindow(win_name, img):
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 32, 32)
    cv2.imshow(win_name, img)


def getDocMask(img):
 
    h, w, c = img.shape

    # img = cv2.GaussianBlur(img, (3, 3), 0)

    mask = np.zeros(img.shape[:2], np.uint8)
    k = min(h//3, w//3)
    mask[:, :] = cv2.GC_PR_BGD
    mask[k//2:h-k//2, k//2:w-k//2] = cv2.GC_PR_FGD
    # mask[h//2-k:h//2+k, w//2-k:w//2+k] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1,65), np.float64)
    fgd_model = np.zeros((1,65), np.float64)

    cv2.grabCut(img, mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)

    result = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    return result

def getDocCorners(img):
    # unused 
    img = img.copy()
    img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_proc = cv2.GaussianBlur(img_proc, (3, 3), 0)

    img_proc = cv2.Canny(img_proc, 350, 500)

    if DEBUG:
        showImgInNewWindow('canny', img_proc)

    contours, hierarchy = cv2.findContours(img_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if DEBUG:
        tmp = img.copy()
        cv2.drawContours(tmp, contours, -1, (0,255,0), 3)
        showImgInNewWindow('contours', tmp)

    cnt = max(contours, key=cv2.contourArea)

    s = cnt.sum(axis=2)

    top_left = cnt[np.argmin(s)][0]
    bottom_right = cnt[np.argmax(s)][0]

    d = np.diff(cnt, axis=2)
    top_right = cnt[np.argmin(d)][0]
    bottom_left = cnt[np.argmax(d)][0]

    if DEBUG:
        tmp = img.copy()
        cv2.circle(tmp, tuple(top_left), 8, (0, 0, 255), -1)
        cv2.circle(tmp, tuple(top_right), 8, (0, 255, 0), -1)
        cv2.circle(tmp, tuple(bottom_right), 8, (255, 0, 255), -1)
        cv2.circle(tmp, tuple(bottom_left), 8, (0, 255, 255), -1)
        showImgInNewWindow('corners', tmp)


    return np.array([top_left, top_right, bottom_right, bottom_left])

def getDocCornersFromMask(img, mask):
    img_proc = mask.copy()
    contours, hierarchy = cv2.findContours(img_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if DEBUG:
        tmp = img.copy()
        cv2.drawContours(tmp, contours, -1, (0,255,0), 3)
        showImgInNewWindow('contours', tmp)

    cnt = max(contours, key=cv2.contourArea)

    s = cnt.sum(axis=2)

    top_left = cnt[np.argmin(s)][0]
    bottom_right = cnt[np.argmax(s)][0]

    d = np.diff(cnt, axis=2)
    top_right = cnt[np.argmin(d)][0]
    bottom_left = cnt[np.argmax(d)][0]

    if DEBUG:
        tmp = img.copy()
        cv2.circle(tmp, tuple(top_left), 8, (0, 0, 255), -1)
        cv2.circle(tmp, tuple(top_right), 8, (0, 255, 0), -1)
        cv2.circle(tmp, tuple(bottom_right), 8, (255, 0, 255), -1)
        cv2.circle(tmp, tuple(bottom_left), 8, (0, 255, 255), -1)
        showImgInNewWindow('corners', tmp)


    return np.array([top_left, top_right, bottom_right, bottom_left])


def alignDoc(img, corners, shape=(700, 500)):

    H, W = shape

    dst_corners = np.array([[0, 0], [W, 0], [W, H], [0, H]])

    h, status = cv2.findHomography(corners, dst_corners)

    return cv2.warpPerspective(img, h, (W, H))

def main():

    img = cv2.imread('data/kart1.jpg')

    print('Processing...')

    if DEBUG:
        showImgInNewWindow('input', img)

    mask = getDocMask(img)

    if DEBUG:
        tmp = cv2.bitwise_and(img, img, mask=mask)
        showImgInNewWindow('masked', tmp)

    corners = getDocCornersFromMask(img, mask)

    output = alignDoc(img, corners)

    print('Done')

    showImgInNewWindow('output', output)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


