import cv2
import os
import csv
import time as t
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from influxdb import InfluxDBClient
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.detection import IRAFStarFinder

detection_frames = 10
seeing_frames = 1000
field_size = 15  #radius in pixels
FWHM = 15  #pixels
TRESHOLD = 5  #in std
METHOD = 'IRAF'  #DAO or IRAF
path_to_cam = 'rtsp://192.168.10.215:554/user=admin&password=&channel=1&stream=0.sdp?'
SHOW_BRITEST_STAR = 'NO'  #YES or NO
WRITE_FITS = 'YES'  #YES or NO

def CalcSeeing(detection_frames, seeing_frames, field_size, FWHM, TRESHOLD, METHOD, path_to_cam, SHOW_BRITEST_STAR, WRITE_FITS):
    cap = cv2.VideoCapture(path_to_cam)    
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)    
    for i in range(1, detection_frames):    
        ret, img = cap.read()
        gray_img = gray_img + cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    cap.release()

    mean, median, std = sigma_clipped_stats(gray_img, sigma=3.0)
    if METHOD == 'DAO':
        daofind = DAOStarFinder(fwhm = FWHM, threshold=TRESHOLD*std)    
        sourses = daofind(gray_img - median)
    elif METHOD == 'IRAF':
        iraffind = IRAFStarFinder(fwhm = FWHM, threshold=TRESHOLD*std)
        sourses = iraffind(gray_img - median)
    else:
        print('Incorrect method of finding stars')
        exit() 
    idx = np.argmax(sourses['flux'])
    measuredstar = [ sourses['xcentroid'][idx], sourses['ycentroid'][idx] ]
    
    if SHOW_BRITEST_STAR == 'YES':
        plt.imshow(gray_img, cmap='Greys', interpolation='nearest')
        plt.scatter(measuredstar[0], measuredstar[1], color='blue')
        plt.show()
    print(measuredstar)

    data = np.zeros([2*field_size, 2*field_size, seeing_frames])
    color_data = np.zeros([2*field_size, 2*field_size, 3, seeing_frames])
    x = np.zeros(seeing_frames)
    y = np.zeros(seeing_frames)

    cap = cv2.VideoCapture(path_to_cam)
    i = 0
    while i < seeing_frames:        
        try:
            ret, img = cap.read()
        except Exception as e:
            cap = cv2.VideoCapture(path_to_cam)
            ret, img = cap.read()
            print('Something went wrong while reading from camera:', e)

        color_data[..., i] = img[int(measuredstar[1])-field_size:int(measuredstar[1])+field_size, int(measuredstar[0])-field_size:int(measuredstar[0])+field_size,:]
        edit_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        edit_img = edit_img[int(measuredstar[1])-field_size:int(measuredstar[1])+field_size, int(measuredstar[0])-field_size:int(measuredstar[0])+field_size]            
        data[..., i] = edit_img
        arr = data[...,i]
        arr = arr - 2*np.median(arr)
        arr[arr<0] = 0
        x[i], y[i] = ndimage.measurements.center_of_mass(arr) 
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            if i%100 == 0:
                print(i)
            i = i + 1
    cap.release()
    
    time = str(datetime.datetime.now().time()) 
    FILENAME = str(datetime.date.today())
    if WRITE_FITS == 'YES':
        hdu = fits.PrimaryHDU(color_data.astype(np.int32))
        hdu.writeto(FILENAME + '_' + time + '.fits')
    print('stage 2 complete')

    for i in range(0, seeing_frames):
        arr = data[...,i]
        arr = arr - 2*np.median(arr)
        arr[arr<0] = 0
        x[i], y[i] = ndimage.measurements.center_of_mass(arr)
    
    k, b = np.polyfit(x, y, deg=1)
    dx = (x[-1] - x[0]) / seeing_frames
    sigma = 0.    
    for i in range(0, seeing_frames):
        sigma = sigma + (x[i] - x[0] - dx*i)**2 + (y[i] - k*(x[0] + dx*i) - b)**2 
    sigma = np.sqrt(sigma/seeing_frames) 
    seeing = 2.83 * np.power((sigma*4.4/206265), 6/5) * np.power( (0.04/(550*1e-9)), 1/5) * 206265
    
    client = InfluxDBClient(host='eagle.sai.msu.ru', port=80)
    reaseeing = client.query(str('SELECT value FROM "massdimm.seeing" WHERE type=\'profile\' AND time > now() - 5m;'),database='taxandria').get_points()  
    
    print('time:', time)    
    print('sigma:', sigma)
    print('seeing:', seeing)
    print('real seeing:', list(reaseeing))
    
    FILENAME = str(datetime.date.today()) + '.csv'
    with open(FILENAME, "a", newline="") as file:
        msg = [time, sigma, seeing, list(reaseeing)]
        writer = csv.writer(file)
        writer.writerow(msg)

    FILENAME = str(datetime.date.today()) + str(time) + 'xy' + '.csv'   
    with open(FILENAME, "a", newline="") as file:
        msg = [x[i], y[i]]
        writer = csv.writer(file)
        writer.writerow(msg)
    print('finish')
    t.sleep(60)
    

while True:
    try:    
        CalcSeeing(detection_frames, seeing_frames, field_size, FWHM, TRESHOLD, METHOD, path_to_cam, SHOW_BRITEST_STAR, WRITE_FITS)
    except Exception as e:
        t.sleep(5)
        print('Something was wrong. Restarting\n Exception:', e)

