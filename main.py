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


if _name_ == "_main_":

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
    img2 = atom(128, 128, 0, 0.1)
    img3 = atom(128, 128, 0.3, 0.3)
    img4 = atom(128, 128, -0.3, 0.1)

    fourier2d(img2, 1)
    fourier2d(img3, 1)
    fourier2d(img4, 1)

    #ne pas oublier de commenter dans la question 7 !

    #b contour !
    imgContour1 = cv.imread(r"diagonal.png")
    plt.figure()
    plt.imshow(imgContour1)
    

    plt.show()
