#!/usr/bin/env python
# coding: utf-8

# # MWA: Lab 05
# ### Identyfikacja tacy i monet
# Inż. Gabriela Szczesna

# #### 1. Import bibliotek

# In[30]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# #### 2. Pobranie obrazów z folderu

# In[31]:


DIRECTORY = 'Coins'
coins_images = []

for entry in os.scandir(DIRECTORY):
    if entry.path.endswith('.jpg') and entry.is_file():
        try:
            img_data = cv2.imread(entry.path)
            coins_images.append(img_data)
        except Exception:
            pass


# #### 3. Wyświetlenie obrazów z użyciem biblioteki matplotlib

# In[32]:


fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.imshow(cv2.cvtColor(coins_images[0], cv2. COLOR_BGR2RGB))
plt.show()


# #### 4. Funkcja znajdująca monety

# In[33]:


def find_coins(image):
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    outline = cv2.Canny(k_open, 100, 100, L2gradient=True)
    coins = cv2.HoughCircles(outline, cv2.HOUGH_GRADIENT, 1.5, minDist=30, minRadius=13, maxRadius=39)
    
    if coins is not None:
        coins = np.round(coins[0,:]).astype("int")
        for (x, y, r) in coins:
            cv2.circle(image, (x, y), r, (0,0,0), 6)
    else:
        print("No circles found")
        
    # posortowanie okręgów po promieniu
    coins = coins[np.argsort(coins[:,2])]
    
    cv2.circle(image, (coins[-1][0], coins[-1][1]), coins[-1][2], (0,0,255), 6)
    cv2.circle(image, (coins[-2][0], coins[-2][1]), coins[-2][2], (0,0,255), 6)
        
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()  


# In[34]:


find_coins(coins_images[0])


# #### 5. Funkcja znajdujące tacę

# In[35]:


def find_tray(image):
    # zmiana na skalę szarości
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pozbycie się zewnętrznych pikseli przez rozmycie
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    # obliczenie zakresu do stworzenia krawędzi na małych obszarach obrazu
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    # operacje morfologiczne
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    # wykrycie krawędzi
    outline = cv2.Canny(k_open, 100, 100, L2gradient=True)
    # wykrycie lini przy użyciu transformaty Hougha
    lines = cv2.HoughLinesP(outline, 1, np.pi/180, 100, minLineLength=50, maxLineGap=20)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(conv_img, (x1, y1), (x2, y2), (0,0,255), 6)
    
        
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(cv2.cvtColor(conv_img, cv2.COLOR_GRAY2RGB))
    plt.show()  


# In[36]:


find_tray(coins_images[1])


# In[ ]:




