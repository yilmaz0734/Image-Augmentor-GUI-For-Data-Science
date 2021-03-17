import PyQt5
import os
import imutils
import cv2
import numpy as np
from PIL import Image as im 
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtGui import QGuiApplication
import sys
#Image augmentation GUI App


def contour_crop_no_resize(image,dim):
    '''
    Contour and crop the image (generally used in brain mri images and object detection)
    :param image: cv2 object
    '''
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def contour_crop_resize(image,dim):
    '''
    Contour and crop the image (generally used in brain mri images and object detection)
    :param image: cv2 object
    '''
    grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    grayscale=cv2.GaussianBlur(grayscale,(5,5),0)
    threshold_image=cv2.threshold(grayscale,45,255,cv2.THRESH_BINARY)[1]
    threshold_image=cv2.erode(threshold_image,None,iterations=2)
    threshold_image=cv2.dilate(threshold_image,None,iterations=2)

    contour=cv2.findContours(threshold_image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=imutils.grab_contours(contour)
    c=max(contour,key=cv2.contourArea)

    extreme_pnts_left=tuple(c[c[:,:,0].argmin()][0])
    extreme_pnts_right=tuple(c[c[:,:,0].argmax()][0])
    extreme_pnts_top=tuple(c[c[:,:,1].argmin()][0])
    extreme_pnts_bot=tuple(c[c[:,:,1].argmax()][0])

    new_image=image[extreme_pnts_top[1]:extreme_pnts_bot[1],extreme_pnts_left[0]:extreme_pnts_right[0]]
    resized = cv2.resize(new_image, dim, interpolation = cv2.INTER_AREA)
    return resized
    

def brightness_increase(image,brightness):
    '''
    increase the brightness of the image
    :param image: cv2 object
    :param brightness: brightness increasing level
    '''
    bright=np.ones(image.shape,dtype="uint8")*brightness
    brightincreased=cv2.add(image,bright)

    return brightincreased

def decrease_brightness(image,brightness):
    '''
    decrease the brightness of the image
    :param image: cv2 object
    :param brightness: brightness decreasing level
    '''
    bright=np.ones(image.shape,dtype="uint8")*50
    brightdecrease=cv2.subtract(image,bright)

    return brightdecrease


def rotate(image,angle=90, scale=1.0):
    '''
    Rotate the image
    :param image: cv2 object
    :param image: image to be processed
    :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
    :param scale: Isotropic scale factor.
    '''
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

def flip(image,axis):
    '''
    Flip the image
    :param image: cv2 object
    :param axis: axis to flip
    '''
    flip=cv2.flip(image,axis)

    return flip

def sharpen(image):
    '''
    Sharpen the image
    :param image: cv2 object
    '''
    sharpening=np.array([ [-1,-1,-1],
                        [-1,10,-1],
                        [-1,-1,-1] ])
    sharpened=cv2.filter2D(image,-1,sharpening)

    return sharpened

def shear(image,axis):
    '''
    Shear the image
    :param image: cv2 object
    :param axis: axis which image will be sheared
    '''

    rows, cols, dim = image.shape
    if axis==0:
        M = np.float32([[1, 0.5, 0],
                        [0, 1  , 0],
                        [0, 0  , 1]])

    elif axis==1:
        M = np.float32([[1,   0, 0],
                        [0.5, 1, 0],
                        [0,   0, 1]])
    sheared_img = cv2.warpPerspective(image,M,(int(cols*1.5),int(rows*1.5)))

    return sheared_img


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('augmentation.ui', self)


        self.show()
        self.setWindowTitle("Image Augmentor")
        mypixmap=QtGui.QPixmap("2582365.ico")
        my_icon=QtGui.QIcon(mypixmap)
        self.setWindowIcon(my_icon)
        self.lineEdit=self.findChild(QtWidgets.QLineEdit,'lineEdit')
        self.checkBox = self.findChild(QtWidgets.QCheckBox,'checkBox_1')
        self.checkBox_2=self.findChild(QtWidgets.QCheckBox,'checkBox_2')
        self.checkBox_3=self.findChild(QtWidgets.QCheckBox,'checkBox_3')
        self.checkBox_4=self.findChild(QtWidgets.QCheckBox,'checkBox_4')
        self.checkBox_5=self.findChild(QtWidgets.QCheckBox,'checkBox_5')
        self.checkBox_6=self.findChild(QtWidgets.QCheckBox,'checkBox_6')
        self.checkBox_7=self.findChild(QtWidgets.QCheckBox,'checkBox_7')
        self.button=self.findChild(QtWidgets.QPushButton,'pushButton')
        self.button2=self.findChild(QtWidgets.QPushButton,'pushButton_2')
        self.button3=self.findChild(QtWidgets.QPushButton,'pushButton_3')
        self.button2.clicked.connect(self.clear)
        self.button3.clicked.connect(self.clearline)
        self.progress=self.findChild(QtWidgets.QProgressBar,'progressBar')

        self.spin1=self.findChild(QtWidgets.QSpinBox,'spinBox')
        self.spin2=self.findChild(QtWidgets.QSpinBox,'spinBox_2')
        self.button.clicked.connect(self.submit)

    def clearline(self):
        self.lineEdit.clear()

    def clear(self):

        if self.button2.text()=="Clear Choices":
            self.button2.setText("Toggle")
            self.checkBox.setChecked(False)
            self.checkBox_2.setChecked(False)
            self.checkBox_3.setChecked(False)
            self.checkBox_4.setChecked(False)
            self.checkBox_5.setChecked(False)
            self.checkBox_6.setChecked(False)
            self.checkBox_7.setChecked(False)
        elif self.button2.text()=="Toggle":
            self.button2.setText("Clear Choices")
            self.checkBox.setChecked(True)
            self.checkBox_2.setChecked(True)
            self.checkBox_3.setChecked(True)
            self.checkBox_4.setChecked(True)
            self.checkBox_5.setChecked(True)
            self.checkBox_6.setChecked(True)
            self.checkBox_7.setChecked(True)
        

    def submit(self):            
        counter=0
        dim=(int(self.spin1.value()),int(self.spin2.value()))
        
        path=self.lineEdit.text()
        if str(path).startswith('"') and str(path).endswith('"'):
            path=path[1:-1]
        my_path=str(path).split("\\")
        folder_name=my_path.copy().pop()
        my_path_popped=my_path[:-1]
        # using list comprehension 
        def listToString(s):  
    
            # initialize an empty string 
            str1 = ""  
            
            # traverse in the string   
            for ele in s:  
                str1 += ele  
                str1 += "/" 
            
            # return string   
            return str1  
        poppedString=listToString(my_path_popped)
   
        if not os.path.exists(poppedString):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText("Change your path!")
            msg.setWindowTitle("Error!")
            msg.setDetailedText("Program could not find the path you provided.")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            msg.exec_()
   
        String=listToString(my_path)
        i=float(0.000001)
        aug_path=poppedString+"/augmented_"+folder_name

        if not os.path.exists(aug_path):
        
            os.makedirs(aug_path)
        
 
        for subdir, dirs, files in os.walk(String):
  
            for file in files:
            
                QGuiApplication.processEvents() 
                 
                filepath = subdir + "/" + file
                imagem=cv2.imread(filepath)
                resized=0
                if self.checkBox.isChecked():
                
                    cv2.imwrite(aug_path+"/cntrdrszd_"+file,contour_crop_resize(imagem,dim))
                    resized=contour_crop_resize(imagem,dim)

                else:
                    
                    cv2.imwrite(aug_path+"/rszd_"+file,contour_crop_no_resize(imagem,dim))
                    resized=contour_crop_no_resize(imagem,dim)
                    
                        
                    


                if self.checkBox_2.isChecked():
                    cv2.imwrite(aug_path+"/brinc_"+file,brightness_increase(resized,50))
            

                if self.checkBox_3.isChecked():
            
                    cv2.imwrite(aug_path+"/brdec_"+file,decrease_brightness(resized,50))
                
                if self.checkBox_4.isChecked():
                
                    cv2.imwrite(aug_path+"/rtd90_"+file,rotate(resized,90,1))
                    cv2.imwrite(aug_path+"/rtd45_"+file,rotate(resized,45,1))
                    cv2.imwrite(aug_path+"/rtd30_"+file,rotate(resized,30,1))
                    cv2.imwrite(aug_path+"/rtd270_"+file,rotate(resized,270,1))
                    cv2.imwrite(aug_path+"/rtd315_"+file,rotate(resized,315,1))
                    cv2.imwrite(aug_path+"/rtd330_"+file,rotate(resized,330,1))

                    if self.checkBox_2.isChecked():
                
                        cv2.imwrite(aug_path+"/binc_rtd90_"+file,rotate(brightness_increase(resized,50),90,1))
                        cv2.imwrite(aug_path+"/binc_rtd45_"+file,rotate(brightness_increase(resized,50),45,1))
                        cv2.imwrite(aug_path+"/binc_rtd30_"+file,rotate(brightness_increase(resized,50),30,1))
                        cv2.imwrite(aug_path+"/binc_rtd270_"+file,rotate(brightness_increase(resized,50),270,1))
                        cv2.imwrite(aug_path+"/binc_rtd315_"+file,rotate(brightness_increase(resized,50),315,1))
                        cv2.imwrite(aug_path+"/binc_rtd330_"+file,rotate(brightness_increase(resized,50),330,1))
                    if self.checkBox_3.isChecked():
                    
                        cv2.imwrite(aug_path+"/bdec_rtd90_"+file,rotate(decrease_brightness(resized,50),90,1))
                        cv2.imwrite(aug_path+"/bdec_rtd45_"+file,rotate(decrease_brightness(resized,50),45,1))
                        cv2.imwrite(aug_path+"/bdec_rtd30_"+file,rotate(decrease_brightness(resized,50),30,1))
                        cv2.imwrite(aug_path+"/bdec_rtd270_"+file,rotate(decrease_brightness(resized,50),270,1))
                        cv2.imwrite(aug_path+"/bdec_rtd315_"+file,rotate(decrease_brightness(resized,50),315,1))
                        cv2.imwrite(aug_path+"/bdec_rtd330_"+file,rotate(decrease_brightness(resized,50),330,1))

                if self.checkBox_5.isChecked():
        
                    cv2.imwrite(aug_path+"/flipxy_"+file,flip(resized,-1))
                    cv2.imwrite(aug_path+"/flipx_"+file,flip(resized,0))
                    cv2.imwrite(aug_path+"/flipy_"+file,flip(resized,1))

                    if self.checkBox_2.isChecked():
            
                        cv2.imwrite(aug_path+"/binc_flipxy_"+file,flip(brightness_increase(resized,50),-1))
                        cv2.imwrite(aug_path+"/binc_flipx_"+file,flip(brightness_increase(resized,50),0))
                        cv2.imwrite(aug_path+"/bdec_flipy_"+file,flip(brightness_increase(resized,50),1))
                    
                    if self.checkBox_3.isChecked():
        
                        cv2.imwrite(aug_path+"/bdec_flipxy_"+file,flip(decrease_brightness(resized,50),-1))
                        cv2.imwrite(aug_path+"/bdec_flipx_"+file,flip(decrease_brightness(resized,50),0))
                        cv2.imwrite(aug_path+"/bdec_flipy_"+file,flip(decrease_brightness(resized,50),1))

                    if self.checkBox_4.isChecked():
        
                        cv2.imwrite(aug_path+"/rtd90_flipxy_"+file,flip(rotate(resized,90,1),-1))
                        cv2.imwrite(aug_path+"/rtd45_flipxy_"+file,flip(rotate(resized,45,1),-1))
                        cv2.imwrite(aug_path+"/rtd30_flipxy_"+file,flip(rotate(resized,30,1),-1))
                        cv2.imwrite(aug_path+"/rtd270_flipxy_"+file,flip(rotate(resized,270,1),-1))
                        cv2.imwrite(aug_path+"/rtd315_flipxy_"+file,flip(rotate(resized,315,1),-1))
                        cv2.imwrite(aug_path+"/rtd330_flipxy_"+file,flip(rotate(resized,330,1),-1))

                        cv2.imwrite(aug_path+"/rtd90_flipx_"+file,flip(rotate(resized,90,1),0))
                        cv2.imwrite(aug_path+"/rtd45_flipx_"+file,flip(rotate(resized,45,1),0))
                        cv2.imwrite(aug_path+"/rtd30_flipx_"+file,flip(rotate(resized,30,1),0))
                        cv2.imwrite(aug_path+"/rtd270_flipx_"+file,flip(rotate(resized,270,1),0))
                        cv2.imwrite(aug_path+"/rtd315_flipx_"+file,flip(rotate(resized,315,1),0))
                        cv2.imwrite(aug_path+"/rtd330_flipx_"+file,flip(rotate(resized,330,1),0))

                        cv2.imwrite(aug_path+"/rtd90_flipy_"+file,flip(rotate(resized,90,1),1))
                        cv2.imwrite(aug_path+"/rtd45_flipy_"+file,flip(rotate(resized,45,1),1))
                        cv2.imwrite(aug_path+"/rtd30_flipy_"+file,flip(rotate(resized,30,1),1))
                        cv2.imwrite(aug_path+"/rtd270_flipy_"+file,flip(rotate(resized,270,1),1))
                        cv2.imwrite(aug_path+"/rtd315_flipy_"+file,flip(rotate(resized,315,1),1))
                        cv2.imwrite(aug_path+"/rtd330_flipy_"+file,flip(rotate(resized,330,1),1))

                    if self.checkBox_4.isChecked() and self.checkBox_2.isChecked():
                    
                        cv2.imwrite(aug_path+"/binc_rtd90_flipxy_"+file,flip(rotate(brightness_increase(resized,50),90,1),-1))
                        cv2.imwrite(aug_path+"/binc_rtd45_flipxy_"+file,flip(rotate(brightness_increase(resized,50),45,1),-1))
                        cv2.imwrite(aug_path+"/binc_rtd30_flipxy_"+file,flip(rotate(brightness_increase(resized,50),30,1),-1))
                        cv2.imwrite(aug_path+"/binc_rtd270_flipxy_"+file,flip(rotate(brightness_increase(resized,50),270,1),-1))
                        cv2.imwrite(aug_path+"/binc_rtd315_flipxy_"+file,flip(rotate(brightness_increase(resized,50),315,1),-1))
                        cv2.imwrite(aug_path+"/binc_rtd330_flipxy_"+file,flip(rotate(brightness_increase(resized,50),330,1),-1))

                        cv2.imwrite(aug_path+"/binc_rtd90_flipx_"+file,flip(rotate(brightness_increase(resized,50),90,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd45_flipx_"+file,flip(rotate(brightness_increase(resized,50),45,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd30_flipx_"+file,flip(rotate(brightness_increase(resized,50),30,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd270_flipx_"+file,flip(rotate(brightness_increase(resized,50),270,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd315_flipx_"+file,flip(rotate(brightness_increase(resized,50),315,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd330_flipx_"+file,flip(rotate(brightness_increase(resized,50),330,1),0))

                        cv2.imwrite(aug_path+"/binc_rtd90_flipy_"+file,flip(rotate(brightness_increase(resized,50),90,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd45_flipy_"+file,flip(rotate(brightness_increase(resized,50),45,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd30_flipy_"+file,flip(rotate(brightness_increase(resized,50),30,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd270_flipy_"+file,flip(rotate(brightness_increase(resized,50),270,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd315_flipy_"+file,flip(rotate(brightness_increase(resized,50),315,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd330_flipy_"+file,flip(rotate(brightness_increase(resized,50),330,1),1))

                    if self.checkBox_4.isChecked() and self.checkBox_3.isChecked():
                
                        cv2.imwrite(aug_path+"/bdec_rtd90_flipxy_"+file,flip(rotate(decrease_brightness(resized,50),90,1),-1))
                        cv2.imwrite(aug_path+"/bdec_rtd45_flipxy_"+file,flip(rotate(decrease_brightness(resized,50),45,1),-1))
                        cv2.imwrite(aug_path+"/bdec_rtd30_flipxy_"+file,flip(rotate(decrease_brightness(resized,50),30,1),-1))
                        cv2.imwrite(aug_path+"/bdec_rtd270_flipxy_"+file,flip(rotate(decrease_brightness(resized,50),270,1),-1))
                        cv2.imwrite(aug_path+"/bdec_rtd315_flipxy_"+file,flip(rotate(decrease_brightness(resized,50),315,1),-1))
                        cv2.imwrite(aug_path+"/bdec_rtd330_flipxy_"+file,flip(rotate(decrease_brightness(resized,50),330,1),-1))

                        cv2.imwrite(aug_path+"/bdec_rtd90_flipx_"+file,flip(rotate(decrease_brightness(resized,50),90,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd45_flipx_"+file,flip(rotate(decrease_brightness(resized,50),45,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd30_flipx_"+file,flip(rotate(decrease_brightness(resized,50),30,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd270_flipx_"+file,flip(rotate(decrease_brightness(resized,50),270,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd315_flipx_"+file,flip(rotate(decrease_brightness(resized,50),315,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd330_flipx_"+file,flip(rotate(decrease_brightness(resized,50),330,1),0))

                        cv2.imwrite(aug_path+"/bdec_rtd90_flipy_"+file,flip(rotate(decrease_brightness(resized,50),90,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd45_flipy_"+file,flip(rotate(decrease_brightness(resized,50),45,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd30_flipy_"+file,flip(rotate(decrease_brightness(resized,50),30,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd270_flipy_"+file,flip(rotate(decrease_brightness(resized,50),270,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd315_flipy_"+file,flip(rotate(decrease_brightness(resized,50),315,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd330_flipy_"+file,flip(rotate(decrease_brightness(resized,50),330,1),1))

                        

                if self.checkBox_6.isChecked():
                
                    cv2.imwrite(aug_path+"/shrpnd_"+file,sharpen(resized))

                    if self.checkBox_2.isChecked():
                
                        cv2.imwrite(aug_path+"/binc_shrpnd_"+file,sharpen(brightness_increase(resized,50)))

                    if self.checkBox_3.isChecked():
                    
                        cv2.imwrite(aug_path+"/bdec_shrpnd_"+file,sharpen(decrease_brightness(resized,50)))

                    if self.checkBox_4.isChecked():
        
                        cv2.imwrite(aug_path+"/rtd90_shrpnd_"+file,sharpen(rotate(resized,90,1)))
                        cv2.imwrite(aug_path+"/rtd45_shrpnd_"+file,sharpen(rotate(resized,45,1)))
                        cv2.imwrite(aug_path+"/rtd30_shrpnd_"+file,sharpen(rotate(resized,30,1)))
                        cv2.imwrite(aug_path+"/rtd270_shrpnd_"+file,sharpen(rotate(resized,270,1)))
                        cv2.imwrite(aug_path+"/rtd315_shrpnd_"+file,sharpen(rotate(resized,315,1)))
                        cv2.imwrite(aug_path+"/rtd330_shrpnd_"+file,sharpen(rotate(resized,330,1)))

                    if self.checkBox_5.isChecked():
        
                        cv2.imwrite(aug_path+"/flipxy_shrpnd_"+file,sharpen(flip(resized,-1)))
                        cv2.imwrite(aug_path+"/flipx_shrpnd_"+file,sharpen(flip(resized,0)))
                        cv2.imwrite(aug_path+"/flipy_shrpnd"+file,sharpen(flip(resized,1)))

                    if self.checkBox_4.isChecked() and self.checkBox_2.isChecked():
                
                        cv2.imwrite(aug_path+"/binc_rtd90_shrpnd_"+file,sharpen(rotate(brightness_increase(resized,50),90,1)))
                        cv2.imwrite(aug_path+"/binc_rtd45_shrpnd_"+file,sharpen(rotate(brightness_increase(resized,50),45,1)))
                        cv2.imwrite(aug_path+"/binc_rtd30_shrpnd_"+file,sharpen(rotate(brightness_increase(resized,50),30,1)))
                        cv2.imwrite(aug_path+"/binc_rtd270_shrpnd_"+file,sharpen(rotate(brightness_increase(resized,50),270,1)))
                        cv2.imwrite(aug_path+"/binc_rtd315_shrpnd_"+file,sharpen(rotate(brightness_increase(resized,50),315,1)))
                        cv2.imwrite(aug_path+"/binc_rtd330_shrpnd_"+file,sharpen(rotate(brightness_increase(resized,50),330,1)))

                    if self.checkBox_4.isChecked() and self.checkBox_3.isChecked():
            
                        cv2.imwrite(aug_path+"/bdec_rtd90_shrpnd_"+file,sharpen(rotate(decrease_brightness(resized,50),90,1)))
                        cv2.imwrite(aug_path+"/bdec_rtd45_shrpnd_"+file,sharpen(rotate(decrease_brightness(resized,50),45,1)))
                        cv2.imwrite(aug_path+"/bdec_rtd30_shrpnd_"+file,sharpen(rotate(decrease_brightness(resized,50),30,1)))
                        cv2.imwrite(aug_path+"/bdec_rtd270_shrpnd_"+file,sharpen(rotate(decrease_brightness(resized,50),270,1)))
                        cv2.imwrite(aug_path+"/bdec_rtd315_shrpnd_"+file,sharpen(rotate(decrease_brightness(resized,50),315,1)))
                        cv2.imwrite(aug_path+"/bdec_rtd330_shrpnd_"+file,sharpen(rotate(decrease_brightness(resized,50),330,1)))

                    if self.checkBox_5.isChecked() and self.checkBox_2.isChecked():
                    
                        cv2.imwrite(aug_path+"/binc_flipxy_shrpnd_"+file,sharpen(flip(brightness_increase(resized,50),-1)))
                        cv2.imwrite(aug_path+"/binc_flipx_shrpnd_"+file,sharpen(flip(brightness_increase(resized,50),0)))
                        cv2.imwrite(aug_path+"/binc_flipy_shrpnd_"+file,sharpen(flip(brightness_increase(resized,50),1)))

                    if self.checkBox_5.isChecked() and self.checkBox_3.isChecked():
                
                        cv2.imwrite(aug_path+"/bdec_flipxy_shrpnd_"+file,sharpen(flip(decrease_brightness(resized,50),-1)))
                        cv2.imwrite(aug_path+"/bdec_flipx_shrpnd_"+file,sharpen(flip(decrease_brightness(resized,50),0)))
                        cv2.imwrite(aug_path+"/bdec_flipy_shrpnd_"+file,sharpen(flip(decrease_brightness(resized,50),1)))

                    if self.checkBox_5.isChecked() and self.checkBox_4.isChecked():
                    
                        cv2.imwrite(aug_path+"/rtd90_flipxy_shrpnd_"+file,sharpen(flip(rotate(resized,90,1),-1)))
                        cv2.imwrite(aug_path+"/rtd45_flipxy_shrpnd_"+file,sharpen(flip(rotate(resized,45,1),-1)))
                        cv2.imwrite(aug_path+"/rtd30_flipxy_shrpnd_"+file,sharpen(flip(rotate(resized,30,1),-1)))
                        cv2.imwrite(aug_path+"/rtd270_flipxy_shrpnd_"+file,sharpen(flip(rotate(resized,270,1),-1)))
                        cv2.imwrite(aug_path+"/rtd315_flipxy_shrpnd_"+file,sharpen(flip(rotate(resized,315,1),-1)))
                        cv2.imwrite(aug_path+"/rtd330_flipxy_shrpnd_"+file,sharpen(flip(rotate(resized,330,1),-1)))

                        cv2.imwrite(aug_path+"/rtd90_flipx_shrpnd_"+file,sharpen(flip(rotate(resized,90,1),0)))
                        cv2.imwrite(aug_path+"/rtd45_flipx_shrpnd_"+file,sharpen(flip(rotate(resized,45,1),0)))
                        cv2.imwrite(aug_path+"/rtd30_flipx_shrpnd_"+file,sharpen(flip(rotate(resized,30,1),0)))
                        cv2.imwrite(aug_path+"/rtd270_flipx_shrpnd_"+file,sharpen(flip(rotate(resized,270,1),0)))
                        cv2.imwrite(aug_path+"/rtd315_flipx_shrpnd_"+file,sharpen(flip(rotate(resized,315,1),0)))
                        cv2.imwrite(aug_path+"/rtd330_flipx_shrpnd_"+file,sharpen(flip(rotate(resized,330,1),0)))

                        cv2.imwrite(aug_path+"/rtd90_flipy_shrpnd_"+file,sharpen(flip(rotate(resized,90,1),1)))
                        cv2.imwrite(aug_path+"/rtd45_flipy_shrpnd_"+file,sharpen(flip(rotate(resized,45,1),1)))
                        cv2.imwrite(aug_path+"/rtd30_flipy_shrpnd_"+file,sharpen(flip(rotate(resized,30,1),1)))
                        cv2.imwrite(aug_path+"/rtd270_flipy_shrpnd_"+file,sharpen(flip(rotate(resized,270,1),1)))
                        cv2.imwrite(aug_path+"/rtd315_flipy_shrpnd_"+file,sharpen(flip(rotate(resized,315,1),1)))
                        cv2.imwrite(aug_path+"/rtd330_flipy_shrpnd_"+file,sharpen(flip(rotate(resized,330,1),1)))


                if self.checkBox_7.isChecked():
                
                    cv2.imwrite(aug_path+"/shrdx_"+file,shear(resized,0))
                    cv2.imwrite(aug_path+"/shrdy_"+file,shear(resized,1))

                    if self.checkBox_2.isChecked():
                        
                        cv2.imwrite(aug_path+"/binc_shrdx_"+file,shear(brightness_increase(resized,50),0))
                        cv2.imwrite(aug_path+"/binc_shrdy_"+file,shear(brightness_increase(resized,50),1))

                    if self.checkBox_3.isChecked():
                        
                        cv2.imwrite(aug_path+"/bdec_shrdx_"+file,shear(decrease_brightness(resized,50),0))
                        cv2.imwrite(aug_path+"/bdec_shrdy_"+file,shear(decrease_brightness(resized,50),1))

                    if self.checkBox_4.isChecked():
                        
                        cv2.imwrite(aug_path+"/rtd90_shrdx_"+file,shear(rotate(resized,90,1),0))
                        cv2.imwrite(aug_path+"/rtd45_shrdx_"+file,shear(rotate(resized,45,1),0))
                        cv2.imwrite(aug_path+"/rtd30_shrdx_"+file,shear(rotate(resized,30,1),0))
                        cv2.imwrite(aug_path+"/rtd270_shrdx_"+file,shear(rotate(resized,270,1),0))
                        cv2.imwrite(aug_path+"/rtd315_shrdx_"+file,shear(rotate(resized,315,1),0))
                        cv2.imwrite(aug_path+"/rtd330_shrdx_"+file,shear(rotate(resized,330,1),0))

                        cv2.imwrite(aug_path+"/rtd90_shrdy_"+file,shear(rotate(resized,90,1),1))
                        cv2.imwrite(aug_path+"/rtd45_shrdy_"+file,shear(rotate(resized,45,1),1))
                        cv2.imwrite(aug_path+"/rtd30_shrdy_"+file,shear(rotate(resized,30,1),1))
                        cv2.imwrite(aug_path+"/rtd270_shrdy_"+file,shear(rotate(resized,270,1),1))
                        cv2.imwrite(aug_path+"/rtd315_shrdy_"+file,shear(rotate(resized,315,1),1))
                        cv2.imwrite(aug_path+"/rtd330_shrdy_"+file,shear(rotate(resized,330,1),1))

                    if self.checkBox_5.isChecked():
                        
                        cv2.imwrite(aug_path+"/flipxy_shrdx_"+file,shear(flip(resized,-1),0))
                        cv2.imwrite(aug_path+"/flipx_shrdx_"+file,shear(flip(resized,0),0))
                        cv2.imwrite(aug_path+"/flipy_shrdx"+file,shear(flip(resized,1),0))

                        cv2.imwrite(aug_path+"/flipxy_shrdy_"+file,shear(flip(resized,-1),1))
                        cv2.imwrite(aug_path+"/flipx_shrdy_"+file,shear(flip(resized,0),1))
                        cv2.imwrite(aug_path+"/flipy_shrdy"+file,shear(flip(resized,1),1))

                    if self.checkBox_6.isChecked():
                    
                        cv2.imwrite(aug_path+"/shrpnd_shrdx"+file,shear(sharpen(resized),0))
                        cv2.imwrite(aug_path+"/shrpnd_shrdy"+file,shear(sharpen(resized),1))

                    if self.checkBox_4.isChecked() and self.checkBox_2.isChecked():
                    
                        cv2.imwrite(aug_path+"/binc_rtd90_shrdx_"+file,shear(rotate(brightness_increase(resized,50),90,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd45_shrdx_"+file,shear(rotate(brightness_increase(resized,50),45,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd30_shrdx_"+file,shear(rotate(brightness_increase(resized,50),30,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd270_shrdx_"+file,shear(rotate(brightness_increase(resized,50),270,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd315_shrdx_"+file,shear(rotate(brightness_increase(resized,50),315,1),0))
                        cv2.imwrite(aug_path+"/binc_rtd330_shrdx_"+file,shear(rotate(brightness_increase(resized,50),330,1),0))

                        cv2.imwrite(aug_path+"/binc_rtd90_shrdy_"+file,shear(rotate(brightness_increase(resized,50),90,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd45_shrdy_"+file,shear(rotate(brightness_increase(resized,50),45,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd30_shrdy_"+file,shear(rotate(brightness_increase(resized,50),30,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd270_shrdy_"+file,shear(rotate(brightness_increase(resized,50),270,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd315_shrdy_"+file,shear(rotate(brightness_increase(resized,50),315,1),1))
                        cv2.imwrite(aug_path+"/binc_rtd330_shrdy_"+file,shear(rotate(brightness_increase(resized,50),330,1),1))

                    if self.checkBox_4.isChecked() and self.checkBox_3.isChecked():
                    
                        cv2.imwrite(aug_path+"/bdec_rtd90_shrdx_"+file,shear(rotate(decrease_brightness(resized,50),90,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd45_shrdx_"+file,shear(rotate(decrease_brightness(resized,50),45,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd30_shrdx_"+file,shear(rotate(decrease_brightness(resized,50),30,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd270_shrdx_"+file,shear(rotate(decrease_brightness(resized,50),270,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd315_shrdx_"+file,shear(rotate(decrease_brightness(resized,50),315,1),0))
                        cv2.imwrite(aug_path+"/bdec_rtd330_shrdx_"+file,shear(rotate(decrease_brightness(resized,50),330,1),0))

                        cv2.imwrite(aug_path+"/bdec_rtd90_shrdy_"+file,shear(rotate(decrease_brightness(resized,50),90,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd45_shrdy_"+file,shear(rotate(decrease_brightness(resized,50),45,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd30_shrdy_"+file,shear(rotate(decrease_brightness(resized,50),30,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd270_shrdy_"+file,shear(rotate(decrease_brightness(resized,50),270,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd315_shrdy_"+file,shear(rotate(decrease_brightness(resized,50),315,1),1))
                        cv2.imwrite(aug_path+"/bdec_rtd330_shrdy_"+file,shear(rotate(decrease_brightness(resized,50),330,1),1))

                    if self.checkBox_5.isChecked() and self.checkBox_2.isChecked():
            
                        cv2.imwrite(aug_path+"/binc_flipxy_shrdx_"+file,shear(flip(brightness_increase(resized,50),-1),0))
                        cv2.imwrite(aug_path+"/binc_flipx_shrdx_"+file,shear(flip(brightness_increase(resized,50),0),0))
                        cv2.imwrite(aug_path+"/binc_flipy_shrdx_"+file,shear(flip(brightness_increase(resized,50),1),0))

                        cv2.imwrite(aug_path+"/binc_flipxy_shrdy_"+file,shear(flip(brightness_increase(resized,50),-1),1))
                        cv2.imwrite(aug_path+"/binc_flipx_shrdy_"+file,shear(flip(brightness_increase(resized,50),0),1))
                        cv2.imwrite(aug_path+"/binc_flipy_shrdy_"+file,shear(flip(brightness_increase(resized,50),1),1))

                    if self.checkBox_5.isChecked() and self.checkBox_3.isChecked():
                
                        cv2.imwrite(aug_path+"/bdec_flipxy_shrdx_"+file,shear(flip(decrease_brightness(resized,50),-1),0))
                        cv2.imwrite(aug_path+"/bdec_flipx_shrdx_"+file,shear(flip(decrease_brightness(resized,50),0),0))
                        cv2.imwrite(aug_path+"/bdec_flipy_shrdx_"+file,shear(flip(decrease_brightness(resized,50),1),0))

                        cv2.imwrite(aug_path+"/bdec_flipxy_shrdy_"+file,shear(flip(decrease_brightness(resized,50),-1),1))
                        cv2.imwrite(aug_path+"/bdec_flipx_shrdy_"+file,shear(flip(decrease_brightness(resized,50),0),1))
                        cv2.imwrite(aug_path+"/bdec_flipy_shrdy_"+file,shear(flip(decrease_brightness(resized,50),1),1))

                    if self.checkBox_5.isChecked() and self.checkBox_4.isChecked():
                
                        cv2.imwrite(aug_path+"/rtd90_flipxy_shrdx_"+file,shear(flip(rotate(resized,90,1),-1),0))
                        cv2.imwrite(aug_path+"/rtd45_flipxy_shrdx_"+file,shear(flip(rotate(resized,45,1),-1),0))
                        cv2.imwrite(aug_path+"/rtd30_flipxy_shrdx_"+file,shear(flip(rotate(resized,30,1),-1),0))
                        cv2.imwrite(aug_path+"/rtd270_flipxy_shrdx_"+file,shear(flip(rotate(resized,270,1),-1),0))
                        cv2.imwrite(aug_path+"/rtd315_flipxy_shrdx_"+file,shear(flip(rotate(resized,315,1),-1),0))
                        cv2.imwrite(aug_path+"/rtd330_flipxy_shrdx_"+file,shear(flip(rotate(resized,330,1),-1),0))

                        cv2.imwrite(aug_path+"/rtd90_flipx_shrdx_"+file,shear(flip(rotate(resized,90,1),0),0))
                        cv2.imwrite(aug_path+"/rtd45_flipx_shrdx_"+file,shear(flip(rotate(resized,45,1),0),0))
                        cv2.imwrite(aug_path+"/rtd30_flipx_shrdx_"+file,shear(flip(rotate(resized,30,1),0),0))
                        cv2.imwrite(aug_path+"/rtd270_flipx_shrdx_"+file,shear(flip(rotate(resized,270,1),0),0))
                        cv2.imwrite(aug_path+"/rtd315_flipx_shrdx_"+file,shear(flip(rotate(resized,315,1),0),0))
                        cv2.imwrite(aug_path+"/rtd330_flipx_shrdx_"+file,shear(flip(rotate(resized,330,1),0),0))

                        cv2.imwrite(aug_path+"/rtd90_flipy_shrdx_"+file,shear(flip(rotate(resized,90,1),1),0))
                        cv2.imwrite(aug_path+"/rtd45_flipy_shrdx_"+file,shear(flip(rotate(resized,45,1),1),0))
                        cv2.imwrite(aug_path+"/rtd30_flipy_shrdx_"+file,shear(flip(rotate(resized,30,1),1),0))
                        cv2.imwrite(aug_path+"/rtd270_flipy_shrdx_"+file,shear(flip(rotate(resized,270,1),1),0))
                        cv2.imwrite(aug_path+"/rtd315_flipy_shrdx_"+file,shear(flip(rotate(resized,315,1),1),0))
                        cv2.imwrite(aug_path+"/rtd330_flipy_shrdx_"+file,shear(flip(rotate(resized,330,1),1),0))



                        cv2.imwrite(aug_path+"/rtd90_flipxy_shrdy_"+file,shear(flip(rotate(resized,90,1),-1),1))
                        cv2.imwrite(aug_path+"/rtd45_flipxy_shrdy_"+file,shear(flip(rotate(resized,45,1),-1),1))
                        cv2.imwrite(aug_path+"/rtd30_flipxy_shrdy_"+file,shear(flip(rotate(resized,30,1),-1),1))
                        cv2.imwrite(aug_path+"/rtd270_flipxy_shrdy_"+file,shear(flip(rotate(resized,270,1),-1),1))
                        cv2.imwrite(aug_path+"/rtd315_flipxy_shrdy_"+file,shear(flip(rotate(resized,315,1),-1),1))
                        cv2.imwrite(aug_path+"/rtd330_flipxy_shrdy_"+file,shear(flip(rotate(resized,330,1),-1),1))

                        cv2.imwrite(aug_path+"/rtd90_flipx_shrdy_"+file,shear(flip(rotate(resized,90,1),0),1))
                        cv2.imwrite(aug_path+"/rtd45_flipx_shrdy_"+file,shear(flip(rotate(resized,45,1),0),1))
                        cv2.imwrite(aug_path+"/rtd30_flipx_shrdy_"+file,shear(flip(rotate(resized,30,1),0),1))
                        cv2.imwrite(aug_path+"/rtd270_flipx_shrdy_"+file,shear(flip(rotate(resized,270,1),0),1))
                        cv2.imwrite(aug_path+"/rtd315_flipx_shrdy_"+file,shear(flip(rotate(resized,315,1),0),1))
                        cv2.imwrite(aug_path+"/rtd330_flipx_shrdy_"+file,shear(flip(rotate(resized,330,1),0),1))

                        cv2.imwrite(aug_path+"/rtd90_flipy_shrdy_"+file,shear(flip(rotate(resized,90,1),1),1))
                        cv2.imwrite(aug_path+"/rtd45_flipy_shrdy_"+file,shear(flip(rotate(resized,45,1),1),1))
                        cv2.imwrite(aug_path+"/rtd30_flipy_shrdy_"+file,shear(flip(rotate(resized,30,1),1),1))
                        cv2.imwrite(aug_path+"/rtd270_flipy_shrdy_"+file,shear(flip(rotate(resized,270,1),1),1))
                        cv2.imwrite(aug_path+"/rtd315_flipy_shrdy_"+file,shear(flip(rotate(resized,315,1),1),1))
                        cv2.imwrite(aug_path+"/rtd330_flipy_shrdy_"+file,shear(flip(rotate(resized,330,1),1),1))
                if self.progress.value!=0 and counter==0:
                    i+=float(float(100)/float(len(files)))
                    self.progress.setValue(i)
                
                print(counter,self.progress.value())
                if 99==self.progress.value() and counter==0:
            
                    self.progress.setValue(0)
                    counter+=1
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("Information")
                    msg.setInformativeText("Completed!")
                    msg.setWindowTitle("Finished")
                    msg.setDetailedText("Your process has been succesfully completed.")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
                    msg.exec_()
                    
                elif 100<=self.progress.value() and counter==0:
                  
                    counter+=1
                    self.progress.setValue(0)
                    msg = QtWidgets.QMessageBox()
                    msg.setIcon(QtWidgets.QMessageBox.Information)
                    msg.setText("Information")
                    msg.setInformativeText("Completed!")
                    msg.setWindowTitle("Finished")
                    msg.setDetailedText("Your process has been succesfully completed.")
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
                    msg.exec_()
                    
                    
                    

app = QtWidgets.QApplication(sys.argv)
app.setStyle('Fusion')
window = Ui()
app.exec_()
