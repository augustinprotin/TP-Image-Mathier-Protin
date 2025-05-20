import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv
from mpl_toolkits import mplot3d

def affiche_image(img):
    plt.figure()
    plt.imshow(img,cmap="gray")
    plt.colorbar()

def  atom(n,m,fx,fy):
    img=np.zeros((n, m))
    x = np.array(np.arange(0,m))
    y = np.arange(0,n)
    e1 = np.exp(1j*2*np.pi*fx*x)
    e2 = np.exp(1j*2*np.pi*fy*y)
    for i in range(n):
        for j in range(m):
            img[i,j] = np.real(e2[i]*np.conjugate(e1[j]))
    return img

def fourier2d(img,fe):
    [height, width] =img.shape

    f = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    n = width/2
    m = height/2

    plt.figure()
    ax = plt.axes(projection='3d')
    x = np.arange(-n/width, n/width, float(fe/width))
    y = np.arange(-m/height, m/height, float(fe/height))
    X, Y = np.meshgrid(x, -y)
    print(X.shape)
    ax.plot_surface(X, Y, np.sqrt(f))
    plt.title({"Spectre - 1"})
    plt.xlabel("Fx")
    plt.ylabel("Fy")

    plt.figure()
    plt.imshow(np.log(5*f+1),extent=[-n/width, n/width, -m/height, m/height])
    plt.colorbar()
    plt.xlabel("Fx")
    plt.ylabel("Fy")
    plt.title("Spectre - 2")


if __name__ == "__main__":

    #1a
    img = atom(128, 128, 0.1, 0)
    plt.figure()
    #2
    plt.imshow(img)
    

    #3 commenter le code de la fonction de fourrier 2d

    #4
    fourier2d(img, 1)
    #commenter l'image 3d et la 2d en corrélation

    #question 6
    '''img2 = atom(128, 128, 0, 0.1)
    img3 = atom(128, 128, 0.3, 0.3)
    img4 = atom(128, 128, -0.3, 0.1)'''

    '''fourier2d(img2, 1)
    fourier2d(img3, 1)
    fourier2d(img4, 1)'''

    #ne pas oublier de commenter dans la question 7 !

    #b contour !
    imgContour1 = cv.imread(r"C:/Users/augus\Documents/Ecole/TP Image/TP-Image-Mathier-Protin/diagonal.png")
    plt.figure()
    plt.imshow(imgContour1)
    

    plt.show()

    # c ne pas oublier de commenter la corrélation des deux spectres !
    # 3 conclure sur la localisation d"une rupture dans le domaine fréquentiel

    # c) textures !
    imgMetal = cv.imread(r"C:/Users/augus\Documents/Ecole/TP Image/TP-Image-Mathier-Protin/Metal0007GP.png")
    plt.figure()
    plt.imshow(imgMetal)
    imgMetal1Grey = cv.cvtColor(imgMetal, cv.COLOR_RGB2GRAY)
    fourier2d(imgMetal1Grey, 1)



    imgWater = cv.imread(r"C:/Users/augus\Documents/Ecole/TP Image/TP-Image-Mathier-Protin/Water0000GP.png")
    plt.figure()
    plt.imshow(imgWater)
    imgWater1Grey = cv.cvtColor(imgWater, cv.COLOR_RGB2GRAY)
    fourier2d(imgWater1Grey, 1)

    imgLeaves = cv.imread(r"C:/Users/augus\Documents/Ecole/TP Image/TP-Image-Mathier-Protin/Leaves0012GP.png")
    plt.figure()
    plt.imshow(imgLeaves)
    imgLeaves1Grey = cv.cvtColor(imgLeaves, cv.COLOR_RGB2GRAY)
    fourier2d(imgLeaves1Grey, 1)

    #enfait ca permet de deviner grossièrement les formes et pattern dans l'image, comme dans le métal on a des lignes horizontales
    # -> on a des lignes verticales, inversémenet pour les lignes verticales.
    #Pour les feuilles, c'est très destructurées l'image de base donc la TF l'est aussi
    #Pour l'eau, c'est destructuré mais pas tant parceque ya quans même des formes horizontales -> formes vertiales dans la tf.


    plt.show()


    #exercice 3, 1)

    imgAtom = atom(128, 128, 0.15, 0.37)
    plt.imshow(imgAtom)



    #2) et 3), binarisation de l'image avec un seuil de 8 en tout cas je crois
    bin_np = (imgAtom >= 0).astype(np.uint8)

    plt.figure()
    plt.imshow(bin_np)

    fe = 1
    #4 transfo de fourrier du bin_np
    #fourier2d(bin_np, fe)

    #5 sous échantillonage avec fe = 0.5
    fourier2d(bin_np, fe/2)

    plt.show()



