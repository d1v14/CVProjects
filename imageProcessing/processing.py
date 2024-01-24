import cv2
import argparse
import matplotlib
from matplotlib import pyplot as plt
import imutils

plt.figure(figsize=(15,10))
def main(source):

    image = cv2.imread(source)

    grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(grayImage,540,550)

    thresh = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY_INV)[1]

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imageWithCountour = image.copy()
    imageWithCountour = cv2.cvtColor(imageWithCountour,cv2.COLOR_BGR2RGB)

    mask = thresh.copy()
    output = cv2.cvtColor(cv2.bitwise_and(image,image,mask=mask),cv2.COLOR_BGR2RGB)
    cv2.drawContours(imageWithCountour,contours,-1,(255,0,0),5)

    plt.subplot(2,3,1),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title("Imported image")

    plt.subplot(2,3,2),plt.imshow(grayImage,cmap="Greys_r")
    plt.title("Gray image")

    plt.subplot(2,3,3),plt.imshow(thresh,cmap="Greys_r")
    plt.title("Thresh image")

    plt.subplot(2,3,4),plt.imshow(imageWithCountour)
    plt.title("Countours on image")

    plt.subplot(2,3,5),plt.imshow(output)
    plt.title("Image without background")

    plt.show()
    cv2.destroyAllWindows()    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing")
    parser.add_argument("source",help="Path to source image")
    args = parser.parse_args()
    main(args.source)