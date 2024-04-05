# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 23:00:45 2022

.bin file analysis for pySPAD and MATLAB GUI
pySPAD DO NOT have ExpIndex,yrange,globalshutter at the first three bytes

@author: Yifang
"""
## .bin file analysis for pySPAD,
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def SPADreadBin(filename,pyGUI=True):
    binfile = open(filename, "rb") #open binfile
    yrange=240
    
    if pyGUI==False:
        '''read first three information and convert to decimal'''
        byte_first3 =binfile.read(3)

        ExpIndex=byte_first3[0] #int type
        yrange=byte_first3[1]
        globalshutter=byte_first3[2]
        print('This Experiment used MATLAB GUI')
        print('ExpIndex is', ExpIndex)
        print('yrange is', yrange)
        print('globalshutter is',globalshutter)
        '''if it is not global shutter fisrt 19200 byte does not count. So read them but not save or convert'''
        rolling_shutter_num= yrange*10*8*(1-globalshutter)*(ExpIndex==1)
        binfile.read(rolling_shutter_num)   
    print ('---Reading SPAD Binary data---')
    spadRange=(yrange,320)#define SPAD sensor size, x=320,y=240
    #spadRange=(320,yrange)#define SPAD sensor size, x=320,y=240
    dtype = np.uint8
    bytedata = np.fromfile(binfile,dtype)
    '''Change bytedata to bits'''
    bytedatasize=len(bytedata) #bytesize=9600*bit plane blocks/frames
    print('bytedatasize is',bytedatasize)
    blocksize=int(bytedatasize/(9600*yrange/240))
    print('blocksize is', blocksize)
    ByteData_bi = np.unpackbits(bytedata,bitorder='little')
    #ByteData_bi = np.unpackbits(bytedata)
    '''Reshape the data to block number*frame number*framesize'''
    datashape=(blocksize,)+spadRange
    BinData=np.reshape(ByteData_bi, datashape)        
    return BinData

''' Bin data shape: [bitplane numbers (block size), 240,320]'''

def countTraceValue (dpath,BinData,xxrange=[10,310],yyrange=[10,230],filename="traceValue.csv"):
    '''set ROI'''
    '''for bulk activity---fibre'''
    # xxrange=[10,310]
    # yyrange=[10,230]
    '''make a mask of ROI, 0---unmask'''
    ROImask=np.ones((240,320))
    ROImask[yyrange[0]:yyrange[1],xxrange[0]:xxrange[1]]=0
    '''photon count sum in each frame, within ROI'''
    blocksize=np.shape(BinData)[0]
    # HotPixelIdx,HotPixelNum=FindHotPixel(BinData,blocksize,thres=0.5)
    BinData=RemoveHotPixelFromTemp(BinData)
    
    print ('blocksize is', blocksize)
    print ('---Calculate trace values----')
    count_value=np.zeros(blocksize)
    for i in range(blocksize):
        frame=BinData[i,:,:]
        frame_mask=np.ma.masked_array(frame, mask=ROImask)
        count_value[i]=frame_mask.sum()
    filename = os.path.join(dpath, filename)
    np.savetxt(filename, count_value, delimiter=",")
    return count_value

def FindHotPixel(dpath,BinData,thres=0.07):
    '''Show the accumulated image'''
    blocksize=np.shape(BinData)[0]
    PixelArrary=np.sum(BinData, axis=0)
    HotPixelIdx=np.argwhere(PixelArrary > thres*blocksize)
    HotPixelNum=len(HotPixelIdx)
    filename = os.path.join(dpath, "HotPixelIdx_YuanPCB.csv")
    np.savetxt(filename, HotPixelIdx, delimiter=",")
    #np.save(filename, HotPixelIdx)
    return HotPixelIdx,HotPixelNum

def RemoveHotPixel(BinData,HotPixelIdx):
    rows, cols = zip(*HotPixelIdx)
    BinData[:, rows, cols] = 0
    return BinData

def RemoveHotPixelFromTemp(BinData):
    current_dir = os.path.dirname(__file__)
    IdxFilename = os.path.join(current_dir, 'HotPixelIdx_YuanPCB.csv')
    
    #IdxFilename="C:/SPAD/HotPixelIdx_YuanPCB.csv"
    #IdxFilename="D:/20220623/HotPixelIdx_MyPCB.csv"
    HotPixelIdx_read=np.genfromtxt(IdxFilename, delimiter=',')
    HotPixelIdx_read=HotPixelIdx_read.astype(int)
    BinData=RemoveHotPixel(BinData,HotPixelIdx_read)
    return BinData


def readMultipleBinfiles(dpath,fileNum,xxRange=[40,200],yyRange=[60,220]):
    for i in range(fileNum):
        Savefilename = "traceValue"+str(i+1)+".csv"
        Binfilename = os.path.join(dpath, "spc_data"+str(i+1)+".bin")
        Bindata=SPADreadBin(Binfilename,pyGUI=False)
        countTraceValue(dpath,Bindata,xxrange=xxRange,yyrange=yyRange,filename=Savefilename) #top green
        #countTraceValue(dpath,Bindata,xxrange=[136,167],yyrange=[151,181],filename=Savefilename) #bottom red
    trace_raw=combineTraces (dpath,fileNum)
        #ShowImage(Bindata,dpath)
    return trace_raw

def readMultipleBinfiles_twoROIs(dpath,fileNum,xxrange_g=[90,210],yyrange_g=[10,110],
                                 xxrange_r=[60,180],yyrange_r=[140,240]):
    #ROI 1
    for i in range(fileNum):
        Savefilename_green = "GreenChannel"+str(i+1)+".csv"
        Savefilename_red = "RedChannel"+str(i+1)+".csv"
        Binfilename = os.path.join(dpath, "spc_data"+str(i+1)+".bin")
        Bindata=SPADreadBin(Binfilename,pyGUI=False)
        countTraceValue(dpath,Bindata,xxrange=xxrange_g,yyrange=yyrange_g,filename=Savefilename_green) #top green
        countTraceValue(dpath,Bindata,xxrange=xxrange_r,yyrange=yyrange_r,filename=Savefilename_red) #bottom red
    
    for i in range(fileNum):
        filename_green = os.path.join(dpath, "GreenChannel"+str(i+1)+".csv")  #csv file is the file contain values for each frame
        filename_red = os.path.join(dpath, "RedChannel"+str(i+1)+".csv")
        print(filename_green)
        if i==0:
            trace_green = np.genfromtxt(filename_green, delimiter=',')
            trace_red = np.genfromtxt(filename_red, delimiter=',')
        else:
            trace_add_green = np.genfromtxt(filename_green, delimiter=',')
            trace_add_red= np.genfromtxt(filename_red, delimiter=',')
            trace_green=np.hstack((trace_green,trace_add_green))
            trace_red =np.hstack((trace_red,trace_add_red))
    filename_green = os.path.join(dpath, "traceGreenAll.csv")
    filename_red = os.path.join(dpath, "traceRedAll.csv")
    np.savetxt(filename_green, trace_green, delimiter=",")
    np.savetxt(filename_red, trace_red, delimiter=",")
    return trace_green,trace_red

def combineTraces (dpath,fileNum):
    for i in range(fileNum):
        filename = os.path.join(dpath, "traceValue"+str(i+1)+".csv")  #csv file is the file contain values for each frame
        print(filename)
        if i==0:
            trace_raw = np.genfromtxt(filename, delimiter=',')
        else:
            trace_add = np.genfromtxt(filename, delimiter=',')
            trace_raw=np.hstack((trace_raw,trace_add))
    filename = os.path.join(dpath, "traceValueAll.csv")
    np.savetxt(filename, trace_raw, delimiter=",")
    return trace_raw

def plot_trace(trace,ax, fs=9938.4, label="trace"):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    ax.plot(taxis,trace,linewidth=1,label=label)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,t)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return ax


def ShowImage(BinData,dpath,xxRange=[0,180],yyRange=[60,240]):
    '''Show the accumulated image'''
    BinData=RemoveHotPixelFromTemp(BinData)
    PixelArrary=np.sum(BinData, axis=0)
    magify=1
    Pixel = (((PixelArrary) / (PixelArrary.max()))*255*magify)
    Pixel = (np.where(Pixel > 255, 255, Pixel)).astype(np.uint8)
    from scipy.ndimage import gaussian_filter
    Pixel_f = gaussian_filter(Pixel, sigma=1)

    img = Image.fromarray(Pixel_f)
    img.show()
    filename = os.path.join(dpath, "FOV_image.png")
    img.save(filename) 
    
    plt.figure(figsize=(8, 8))
    plt.imshow(PixelArrary, cmap='gray')
    plt.colorbar(label='Photon count')
    plt.title('Image with Selected Region')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    # Add a rectangle around the selected region
    rect = patches.Rectangle((xxRange[0], yyRange[0]), xxRange[1]-xxRange[0], yyRange[1]-yyRange[0], 
                              linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    filename = os.path.join(dpath, "FOV_image_ROI.png")
    plt.savefig(filename)

    return img

def find_bright_area(image):
    import cv2
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)
    # Create an empty mask for the contour
    contour_mask = np.zeros_like(gray)
    # Draw the contour on the mask
    cv2.drawContours(contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    # Draw the outline of the contour in red
    cv2.drawContours(image, [max_contour], -1, (0, 0, 255), 2)
    return contour_mask, image

def ShowImage_backgroundRemoved(BinData,BinData_b,dpath):
    '''Show the accumulated image'''
    #BinData=RemoveHotPixelFromTemp(BinData)
    PixelArrary=np.sum(BinData, axis=0)-np.sum(BinData_b, axis=0)
    magify=2.5
    Pixel = (((PixelArrary) / (PixelArrary.max()))*255*magify)
    Pixel = (np.where(Pixel > 255, 255, Pixel)).astype(np.uint8)
    from scipy.ndimage import gaussian_filter
    Pixel_f = gaussian_filter(Pixel, sigma=1.2)
    img = Image.fromarray(Pixel_f)
    img.show()
    filename = os.path.join(dpath, "FOV_image.png")
    img.save(filename) 
    return img
