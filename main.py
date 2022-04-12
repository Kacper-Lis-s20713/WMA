import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    DIRECTORY = 'Coins'
    coins_images = []

    for entry in os.scandir(DIRECTORY):
        if entry.path.endswith('.jpg') and entry.is_file():
            try:
                img_data = cv2.imread(entry.path)
                coins_images.append(img_data)
            except Exception:
                pass

    i = 0
    for coin in coins_images:
        i += 1
        image, coins = find_coins(coin)
        tray = find_tray(image)
        number_of_coins, not_on_tray = count(tray, coins)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        print(f'Taca numer {i} ma {number_of_coins}, poza nią jes {not_on_tray}')


def find_coins(image):
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    outline = cv2.Canny(k_open, 100, 200, L2gradient=True, apertureSize=3)
    coins = cv2.HoughCircles(outline, cv2.HOUGH_GRADIENT, 2, minDist=45, minRadius=15, maxRadius=38,
                             param1=500, param2=100)

    if coins is not None:
        coins = np.round(coins[0, :]).astype("int")
        for (x, y, r) in coins:
            cv2.circle(image, (x, y), r, (0, 0, 0), 6)
    else:
        print("No circles found")

    return image, coins


def find_tray(image):
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    outline = cv2.Canny(k_open, 100, 200, L2gradient=True, apertureSize=3)
    lines = cv2.HoughLinesP(outline, 1, np.pi/180, 120, minLineLength=70, maxLineGap=20)

    x_lu = 0
    y_lu = 0
    x_rd, y_rd = image.shape[0], image.shape[1]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1>x2:
            if x1>x_lu:x_lu = x1
        elif x2>x_lu:x_lu = x2

        if x1<x2:
            if x1<x_rd:x_rd = x1
        elif x2<x_rd:x_rd = x2

        if y1>y2:
            if y1>y_lu:y_lu = y1
        elif y2>y_lu:y_lu = y2

        if y1<y2:
            if y1<y_rd:y_rd = y1
        elif y2<y_rd:y_rd = y2

    # print(f'{x_lu} {y_lu} {x_rd} {y_rd}')
    return x_lu, y_lu, x_rd, y_rd

def count(tray, coins):
    x_lu, y_lu, x_rd, y_rd = tray
    n_coins_tray = 0
    for c in coins:
        if c[0] in range(x_rd, x_lu):
            if c[1] in range(y_rd, y_lu):
                n_coins_tray += 1

    not_on_tray = len(coins) - n_coins_tray
    return n_coins_tray, not_on_tray



if __name__ == '__main__':
    main()