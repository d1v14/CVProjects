import cv2 
import numpy as numpy
from matplotlib import pyplot as plt

img = cv2.imread("./firstStepsInCV/pictures/semyon.jpg")
assert img is not None, "couldnt open file"
img = cv2.resize(img,(1000,1500))

temp = cv2.imread("./firstStepsInCV/pictures/tongue.png")
assert temp is not None, "couldnt open file"
temp = cv2.resize(temp,(100,100))


tempWidth, tempHeight = temp.shape[0:2]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

plt.figure(figsize=(20, 18))

for idx, meth in enumerate(methods, 1):
    img2 = img.copy()
    print(meth)
    method = eval(meth)

    # Применение метода сопоставления шаблона
    res = cv2.matchTemplate(img2, temp, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Выбор координат для рисования прямоугольника
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + tempWidth, top_left[1] + tempHeight)

    # Рисование красного прямоугольника вокруг найденного объекта
    cv2.rectangle(img2, top_left, (top_left[0]+tempWidth,top_left[1]+tempHeight), (0, 0, 255), 2)

    # Вывод результатов
    plt.subplot(3, 4, idx*2-1), plt.imshow(res, cmap='viridis')
    plt.title(meth.split('.')[-1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 4, idx*2), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Detected Object'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

