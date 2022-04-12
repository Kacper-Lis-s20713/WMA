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

    for coin in coins_images:
        image = find_coins(coin)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        break


def find_coins(image):
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    outline = cv2.Canny(k_open, 100, 100, L2gradient=True, apertureSize=7)
    coins = cv2.HoughCircles(outline, cv2.HOUGH_GRADIENT, 2, minDist=25, minRadius=10, maxRadius=34, param1=200, param2=100)

    if coins is not None:
        coins = np.round(coins[0, :]).astype("int")
        for (x, y, r) in coins:
            cv2.circle(image, (x, y), r, (0, 0, 0), 6)
    else:
        print("No circles found")

    return image

if __name__ == '__main__':
    main()