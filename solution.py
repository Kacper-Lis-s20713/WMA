import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2


def main(args):
    """
    Method responsible for executing main loop of program
    :param args: script argument parser
    """
    coins_images = []
    for entry in os.scandir(args.input):
        if entry.path.endswith('.jpg') and entry.is_file():
            img_data = cv2.imread(entry.path)
            coins_images.append((entry.name, img_data))

    for coin_image in coins_images:
        image_name, image = coin_image
        tray_coords = find_tray(image)
        coins = find_coins(image)
        result_info = ''
        if coins is not None:
            result_info, boundary_radius = count_coins(coins, tray_coords) #threshold=0.08
            image = draw_coins(image, coins, boundary_radius, color1=(10, 50, 245), color2=(41, 188, 180))
        result_info = f"Wymiary tacy: " \
                      f"{tray_coords[0] - tray_coords[2]}x{tray_coords[1] - tray_coords[3]}\n" \
                      f"{result_info}"
        image = draw_tray(image, tray_coords, (188, 41, 180))
        show_image(image, image_name, result_info)


def find_coins(image):
    """
    Method responsible for finding coins on image
    :param image: image of coins
    """
    outline = get_outline(image)
    coins = cv2.HoughCircles(outline, cv2.HOUGH_GRADIENT, 1.9, param1=195, param2=95,
                             minDist=26, minRadius=13, maxRadius=39)

    if coins is not None:
        coins = np.round(coins[0, :]).astype("int")
        coins = coins[np.argsort(coins[:, 2])]
    else:
        print("No coins found")

    return coins


def find_tray(image):
    """
    Method responsible for finding coordinates of tray on image
    :param image: image of coins
    """
    outline = get_outline(image)
    lines = cv2.HoughLinesP(outline, 1, np.pi / 180, 130, minLineLength=60, maxLineGap=20)

    x_max, y_max, x_min, y_min = -1, -1, image.shape[1], image.shape[0]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_max = max(x1, x2, x_max)
            y_max = max(y1, y2, y_max)
            x_min = min(x1, x2, x_min)
            y_min = min(y1, y2, y_min)
    else:
        print('no tray found')

    return x_max, y_max, x_min, y_min


def get_outline(image):
    """
    Method responsible for getting edges of objects in image
    :param image: image of coins
    :return: list of edges coordinates
    """
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 3)
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    return cv2.Canny(k_open, 100, 100, L2gradient=True)


def count_coins(coins, tray_coords, threshold=None):
    (x_max, y_max, x_min, y_min) = tray_coords
    coins = coins.tolist()
    coins_5pln = []
    coins_5gr = []

    if threshold is not None:
        for coin_index in range(len(coins) - 1):
            if coins[coin_index][2] * (1.0 + threshold) < coins[coin_index + 1][2]:
                coins_5pln = [coin for coin in coins if coins.index(coin) > coin_index]
                coins_5gr = [coin for coin in coins if coin not in coins_5pln]
                break

        if len(coins_5pln) == 0:
            coins_5gr = coins
    else:
        coins_5pln = coins[-2:]
        coins_5gr = coins[:-2]

    coins_5pln_tray = []
    coins_5gr_tray = []

    for coin in coins_5pln:
        if coin[0] in range(x_min, x_max) and coin[1] in range(y_min, y_max):
            coins_5pln_tray.append(coin)

    for coin in coins_5gr:
        if coin[0] in range(x_min, x_max) and coin[1] in range(y_min, y_max):
            coins_5gr_tray.append(coin)

    coins_5pln = [coin for coin in coins_5pln if coin not in coins_5pln_tray]
    coins_5gr = [coin for coin in coins_5gr if coin not in coins_5gr_tray]

    n_coins_on_tray = round(len(coins_5pln_tray) + len(coins_5gr_tray), 2)
    v_coins_on_tray = round(len(coins_5pln_tray) * 5 + len(coins_5gr_tray) * 0.05, 2)
    n_coins_not_tray = round(len(coins_5pln) + len(coins_5gr), 2)
    v_coins_not_tray = round(len(coins_5pln) * 5 + len(coins_5gr) * 0.05, 2)

    boundary_radius = coins[-1][2] + 1
    if coins_5pln:
        boundary_radius = min(coins_5pln[0][2], boundary_radius)
    if coins_5pln_tray:
        boundary_radius = min(coins_5pln_tray[0][2], boundary_radius)

    return f"Na tacy znajduje się {n_coins_on_tray} monet o wartości {v_coins_on_tray}pln." \
           f" Poza nią znajduje się {n_coins_not_tray} monet o wartości {v_coins_not_tray}pln.",\
           boundary_radius


def draw_tray(image, tray_coords, color=(0, 0, 0), brush_size=6):
    """
    Method responsible for drawing tray border
    :param image: image
    :param tray_coords: coords of tray border
    :param color: color of brush
    :param brush_size: size of brush
    :return: processed image
    """
    (x1, y1, x2, y2) = tray_coords
    cv2.rectangle(image, (x1, y1), (x2, y2), color, brush_size)
    return image


def draw_coins(image, coins, boundary_radius, color1=None, color2=None, brush_size=3):
    """
    Method responsible for drawing coin borders
    :param color2: color for bigger coin
    :param color1: color for smaller coins
    :param boundary_radius: boundary for defining bigger and smaller coins
    :param image: image
    :param coins: array of coins (x,y,r)
    :param brush_size: size of brush
    :return: processed image
    """
    for (coin_x, coin_y, coin_r) in coins:
        if coin_r >= boundary_radius:
            color = color1
        else:
            color = color2
        cv2.circle(image, (coin_x, coin_y), coin_r, color, brush_size)
    return image


def show_image(image, title=None, sub_title=None):
    """
    Method responsible for displaying image
    :param sub_title: information about tray and coins
    :param title: information about coins
    :param image: image of coins
    """
    fig = plt.figure(figsize=(10, 10))
    if title is not None:
        fig.suptitle(title, fontsize=16)
    ax = fig.gca()
    if sub_title is not None:
        ax.set_title(sub_title)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def parse_args():
    """
    Method responsible for creating script arguments
    :return: script argument parser
    """
    parser = argparse.ArgumentParser(description='Path to folder with coin images.')
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='Path to folder with coin images.')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
