from PIL import Image
import numpy as np
import cv2
import math 

def gaussian_filter(sig):
    s = int(np.ceil(3 * sig)) # values upto  3sigma
    y = np.linspace(-s,s, 2*s + 1, dtype=float) # creating a one dim from +3sigma to -3sigma
    K = 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-y**2 / (2 * sig**2))  #applying gaussian to all the values
    K = K / np.sum(K) # make sure the sum of weights in kernel = 1
    return K

def D_Y(): #computing derivative along y axis giving horizontal edges
    F1=gaussian_filter(3)
    F1=np.array(F1)[np.newaxis]
    d1=np.array([[1],[0],[-1]])
    return np.dot(d1,F1)

def D_X():# computing derivative along y axis giving vertical edges
    F2=gaussian_filter(3)
    F2=np.array(F2)[np.newaxis]
    d2=np.array([[1,0,-1]])
    return np.dot(F2.T,d2)

def convolute(I,F,stride=1):
    out_s1=int(((I.shape[0]-F.shape[0])/stride)+1)
    out_s2=int(((I.shape[1]-F.shape[1])/stride)+1)
    out=np.zeros([out_s1,out_s2], dtype = int)
    x=0
    for con1 in range(out_s1): #convolution along the column
        for con in range(out_s2): #convolution along the row
            for row in range(con1,con1+F.shape[0]):
                for col in range(con,con+F.shape[1]):
                    x+=I[row][col]*F[row-con1][col-con]
            out[con1][con]=x
            x=0
    return out


def P(Image,pad): #complete row and column Padding
    out_s1=Image.shape[0]+pad*2
    out_s2=Image.shape[1]+pad*2
    out=np.zeros([out_s1,out_s2], dtype = int)
    for i in range(pad,out_s1-pad):
        for j in range(pad,out_s2-pad):
            out[i][j]=Image[i-pad][j-pad]
    return out

def column_P(Image,pad):
    out=[]
    zeros=np.zeros((1,pad))
    #print(zeros)
    for i in range(len(Image)):
        x=np.hstack((Image[i],zeros[0]))
        out.append(x)
    return np.array(out)

def row_P(Image,pad):
    out=[]
    zeros=np.zeros((1,Image.shape[1]))
    x=Image
    for i in range(pad):
        x=np.vstack((x,zeros))
    out.append(x)
    return np.array(out[0])

def thresh(dir): # thresholding the direction to one of the 4 directions 0,45,90,135
    if dir<0:
        dir+=180
    if 0<= dir <22.5 or 157.5 <= dir<=180:
        return 0
    elif 22.5 <= dir < 67.5:
        return 1
    elif 67.5 <= dir < 112.5:
        return 2
    elif 112.5 <= dir < 157.5:
        return 3

def Magnitude(Img1,Img2): #calculate the magnitude of edge response
    if Img1.shape[0]>Img2.shape[0]: #Padding to make the shape of 2 matrices same
        pad_row=Img1.shape[0]-Img2.shape[0]
        Img2=row_P(Img2,pad_row)
    if Img2.shape[0]>Img1.shape[0]:
        pad_row=Img2.shape[0]-Img1.shape[0]
        Img1=row_P(Img1,pad_row)
    if Img1.shape[1]>Img2.shape[1]:
        pad_col=Img1.shape[1]-Img2.shape[1]
        Img2=column_P(Img2,pad_col)
    if Img2.shape[1]>Img1.shape[1]:
        pad_col=Img2.shape[1]-Img1.shape[1]
        Img1=column_P(Img1,pad_col)
    out_m=np.empty([Img1.shape[0],Img1.shape[1]], dtype = int)
    out_d=np.empty([Img1.shape[0],Img1.shape[1]], dtype = int)
    for i in range(0,Img1.shape[0]):
        for j in range(0,Img1.shape[1]):
            mag=math.sqrt(math.pow(Img1[i][j],2)+math.pow(Img2[i][j],2)) #magnitude of each pixel
            dir=math.atan2(Img2[i][j],Img1[i][j]) #direction of gradients in radians
            dir=dir*180/math.pi
            dir=thresh(dir)
            out_m[i][j],out_d[i][j]=mag,dir
    return out_m,out_d

def Threshold(Image):
    High_thresh=Image.max()*0.3
    low_thresh=High_thresh*0.05
    print(High_thresh,low_thresh)
    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            if Image[i][j]>High_thresh:
                Image[i][j]=255
            elif low_thresh< Image[i][j] <= High_thresh: #performing connected component analysis in here 
                if Image[i-1][j-1]>High_thresh or Image[i-1][j]>High_thresh or Image[i-1][j+1]>High_thresh or Image[i][j-1]>High_thresh or Image[i][j+1]>High_thresh or Image[i+1][j-1]>High_thresh or Image[i+1][j+1]>High_thresh or Image[i+1][j]>High_thresh:
                    Image[i][j]=255
            elif Image[i][j]<low_thresh:
                Image[i][j]=0
    return Image


def NMS(Im1,Im2):
    I1=P(Im1,1) # Padding the image to make the pixel comparison easier
    I2=P(Im2,1)
    #print(Im1.shape,I1.shape)
    #print(Im2.shape,I2.shape)
    out=np.zeros([I1.shape[0],I1.shape[1]], dtype = int)
    for i in range(1,I1.shape[0]-1):
        for j in range(1,I1.shape[1]-1):
            if I2[i][j]==0: # comparing pixels depending upon its gradient direction
                if I1[i][j] > I1[i][j-1] and I1[i][j] > I1[i][j+1]:
                    out[i][j]=I1[i][j]
            elif I2[i][j]==1:
                if I1[i][j] > I1[i-1][j+1] and I1[i][j] > I1[i+1][j-1]:
                    out[i][j]=I1[i][j]
            elif I2[i][j]==2:
                if I1[i][j] > I1[i-1][j] and I1[i][j] > I1[i+1][j]:
                    out[i][j]=I1[i][j]
            elif I2[i][j]==3:
                if I1[i][j] > I1[i-1][j-1] and I1[i][j] > I1[i+1][j+1]:
                    out[i][j]=I1[i][j]
    cv2.imwrite("out_nms.jpg",np.uint8(out))
    threshold=Threshold(out)
    return out,threshold


if __name__=='__main__':

    #reading input image
    I=Image.open("boat.jpg") 
    G_F=gaussian_filter(0.2) #gaussian filter for image smoothing
    G_F=np.array(G_F)[np.newaxis]
    
    #derivative of gaussian along x along y direction
    G_Fx=D_X() 
    G_Fy=D_Y()

    #Smoothing of imaging in x and y direction 
    X=convolute(np.array(I),G_F) 
    Ix=np.uint8(X)
    Y=convolute(np.array(I),G_F.T)
    Iy=np.uint8(Y)
    cv2.imwrite("Ix.jpg",Ix)
    cv2.imwrite("Iy.jpg",Iy)

    #convolve to get edge detection along x and y direction
    x_dash= convolute(X,G_Fx) 
    y_dash= convolute(Y,G_Fy)
    Ix_dash=np.uint8(x_dash) 
    Iy_dash=np.uint8(y_dash)
    
    #saving the image
    cv2.imwrite("Ix_dash.jpg",Ix_dash)
    cv2.imwrite("Iy_dash.jpg",Iy_dash)

    #nms filtering for extra thick edges removal
    Mag,dir=Magnitude(x_dash,y_dash)
    nms_out,threshold=NMS(Mag,dir) 
    cv2.imwrite("Mxy.jpg",np.uint8(Mag))
    cv2.imwrite("Thresh.jpg",np.uint8(threshold))