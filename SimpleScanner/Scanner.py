import argparse
import cv2
import imutils
import matplotlib 
import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize = (20,18))

def main(source):
    # Читаем изображение
    original = cv2.imread(source)
    assert original is not None, "Couldnt open image file"

    copy = original.copy()
    # Изменяем размер изображения, чтобы ускорить работу поиска углов и контуров
    copy = imutils.resize(original,height=500)
    # Сохраняем масштаб исходного изображения, чтобы масштабировать будущий контур
    ratio = original.shape[0]/500.0
    # Переводим изображение в оттенки серого
    copyGray = cv2.cvtColor(copy,cv2.COLOR_BGR2GRAY)
    # Размываем изображение для более успешного поиска контуров
    copyGray = cv2.GaussianBlur(copyGray,(7,7),0)
    # Выделяем углы
    edged = cv2.Canny(copyGray,75,200)
    # Выполняем поиск контуров
    Contours, _ = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # Поскольку мы ищем углы у чека, а чек это прямоугольный лист, нам необходимо найти самый длинный контур, контур формы чека
    # Сортируем найденные контуры по длине контура в обратном порядке,
    Contours = sorted(Contours,key = cv2.contourArea,reverse=True)
    resultCont = list()

    for cont in Contours:
        peri = cv2.arcLength(cont, True)
        # Если контур состоит боллее чем из 4 точек, апроксимируем его, пока не получим 4 точки.
        approxedCont = cv2.approxPolyDP(cont,peri*0.02,True)
        if(len(approxedCont)==4):
            resultCont.append(approxedCont)
            break
    CopyWithApproxCont = copy.copy()
    CopyWithCont = copy.copy()
    cv2.drawContours(CopyWithApproxCont,resultCont,-1,(0,0,255),2)
    cv2.drawContours(CopyWithCont,Contours,-1,(0,0,255),2)
    # Исходный массив точек контура
    npConts = np.array(resultCont)
    # Формируем нормальный двумерный массив и умножаем на масштаб
    npConts = npConts.reshape(4,2) * ratio
    summary = npConts.sum(axis=1)
    # Перекладываем точки массива в правильную последовательность
    # Последовательной левый верх - правый верх - правый низ - левый низ
    rightSeqOfPoints = np.zeros((4,2),dtype="float32")
    rightSeqOfPoints[0] = npConts[np.argmin(summary)]
    rightSeqOfPoints[2] = npConts[np.argmax(summary)]
    differed = np.diff(npConts,axis=1)
    rightSeqOfPoints[3] = npConts[np.argmax(differed)]
    rightSeqOfPoints[1] = npConts[np.argmin(differed)]
    tl , tr , bl , br = rightSeqOfPoints
    #Вычисляем высоту и ширину с каждой из сторон контура, выбираем наибольшую
    topWidth = np.sqrt((tr[0]-tl[0])**2 +(tr[1]-tl[1])**2)
    bottomWidth = np.sqrt((br[0]-bl[0])**2 +(br[1]-bl[1])**2)

    leftHeight = np.sqrt((bl[0]-tl[0])**2 +(bl[1]-tl[1])**2)
    rightHeight = np.sqrt((br[0]-tr[0])**2 +(br[1]-tr[1])**2)

    width , height = max(int(topWidth),int(bottomWidth)) , max(int(leftHeight),int(rightHeight))
    
    dst = np.array([
        [0,0],
        [width-1,0],
        [width-1,height-1],
        [0,height-1]
    ],
        dtype="float32")
    # Вычисляем матрицу поворота для перевода контуров из состояния на фото в вертикальное состояние
    M = cv2.getPerspectiveTransform(rightSeqOfPoints,dst)
   
    scan = original.copy()
    # Поворачиваем и устанавливаем необходимый размер
    scan = cv2.warpPerspective(scan,M,(width,height))
    scan = cv2.cvtColor(scan,cv2.COLOR_BGR2GRAY)

    # Разные виды сканов
    T = cv2.threshold(scan,190,255,cv2.THRESH_BINARY)[1]
    scanBlack = (scan).astype("uint8")*255 
    scanSharper = T
    scanDefault = scan 
    
    # Выводим графики
    
    plt.subplot(3,3,1)
    plt.imshow(cv2.cvtColor(original,cv2.COLOR_BGR2RGB))
    plt.title("Original document")

    plt.subplot(3,3,2)
    plt.imshow(cv2.cvtColor(copy,cv2.COLOR_BGR2RGB))
    plt.title("Copy resized")

    plt.subplot(3,3,3)
    plt.imshow(copyGray,cmap="Greys_r")
    plt.title("Grayscale copy with blur")

    plt.subplot(3,3,4)
    plt.imshow(edged,cmap="Greys_r")
    plt.title("Edged grayscale copy with blur")

    plt.subplot(3,3,5)
    plt.imshow(cv2.cvtColor(CopyWithCont,cv2.COLOR_BGR2RGB))
    plt.title("Resized copy with contours")

    plt.subplot(3,3,6)
    plt.imshow(cv2.cvtColor(CopyWithApproxCont,cv2.COLOR_BGR2RGB))
    plt.title("Resized copy with aproximated contours")
    
    plt.subplot(3,3,7)
    plt.imshow(scanBlack,cmap="Greys_r")
    plt.title("Scan black")

    plt.subplot(3,3,8)
    plt.imshow(scan,cmap="Greys_r")
    plt.title("Scan white")
   

    plt.subplot(3,3,9)
    plt.imshow(scanSharper,cmap="Greys_r")
    plt.title("Contrast scan")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Scanner")
    parser.add_argument("source",help="Path to source image to scan")
    args = parser.parse_args()
    main(args.source)