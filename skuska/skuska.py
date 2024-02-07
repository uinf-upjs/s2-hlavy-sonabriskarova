import cv2 as cv
import numpy as np

#img1 = cv.imread('obr2.jpg', cv.COLOR_BGR2RGB)                   #ked je to potrebne
img1 = cv.imread("out_obr3.jpg")

#skalovanie
print(img1.shape[1])
print(img1.shape[0])

resized_img = cv.resize(img1, (390, 200))
img1 = resized_img
cv.imwrite('obr3_res.jpg', img1)

denoised_img1 = cv.medianBlur(img1, 5)
#denoised_img1 = cv.GaussianBlur(img1, (5, 5), 0)

gray1 = cv.cvtColor(denoised_img1, cv.COLOR_BGR2GRAY)

#img1 = cv.equalizeHist(gray1)

#laplacian
laplacian = cv.Laplacian(gray1,cv.CV_8UC1, ksize=3)
cv.imwrite('obr3_laplacian.jpg', laplacian)

#edges = cv.Canny(img1, 120, 110)
#cv.imwrite('obr3_edg.jpg', edges)

circles1 = cv.HoughCircles(laplacian, cv.HOUGH_GRADIENT, dp=1, minDist=10, param1=10, param2=35, minRadius=10, maxRadius=40)
if circles1 is not None:
        circles1 = np.uint16(np.around(circles1))
        for i in circles1[0, :]:
            cv.circle(img1, (i[0], i[1]), i[2], (0, 255, 0), 2)
cv.imwrite('obr3_final1.jpg', img1)




#obr 2 dava najlepsie vysledky s klasickym canny a ekvalizovanym histogramom

#obr 3 lepsie detekuje s laplacianom + ked obrazok zmensime v smere y-ovej osi, hlavy maju viac tvar kruhu

