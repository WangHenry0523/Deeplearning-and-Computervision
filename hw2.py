import os
import random
import sys
from torch.utils.data import Dataset
from PyQt5 import QtCore, QtWidgets
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torchvision.models as models
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtGui import *
import torch.nn as nn
import torch.nn.functional as F
import torch,torchvision
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic, QtWidgets
from PIL import Image
from PyQt5.QtCore import QBuffer
import io
from torchsummary import summary
import torchvision.transforms as transforms
qtCreatorFile = "test1.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyDraw(QtWidgets.QWidget, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.pix = QPixmap(528, 256)
        self.pix.fill(QColor(0, 0, 0))

        self.endPoint = QPoint()
        self.lastPoint = QPoint()
        self.painter = QPainter()
        self.offset=QPoint(730,170)
        self.loadimage1.clicked.connect(self.loadimage1_clicked)
        self.drawcontour.clicked.connect(self.drawcontour_clicked)
        self.countcoins.clicked.connect(self.countcoins_clicked)
        self.histogramequalization.clicked.connect(self.histogramequalization_clicked)
        self.opening.clicked.connect(self.opening_clicked)
        self.closing.clicked.connect(self.closing_clicked)
        self.showmodelvgg.clicked.connect(self.showmodelvgg_clicked)
        self.accuracyandloss.clicked.connect(self.accuracyandloss_clicked)
        self.reset.clicked.connect(self.reset_clicked)
        self.predict.clicked.connect(self.predict_clicked)
        self.loadimage2.clicked.connect(self.loadimage2_clicked)
        self.showimages.clicked.connect(self.showimages_clicked)
        self.showmodelresnet50.clicked.connect(self.showmodelresnet50_clicked)
        self.comparison.clicked.connect(self.comparison_clicked)
        self.inference.clicked.connect(self.inference_clicked)
        
    def paintEvent(self, event):
        self.painter.begin(self)
        x = 730
        y = 170
        self.painter.drawPixmap(x, y, self.pix)
        self.painter.end()

    def mousePressEvent(self, event):       
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()-self.offset
            self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):        
        if event.buttons() == Qt.LeftButton:       
            self.endPoint = event.pos()-self.offset          
            self.painter.begin(self.pix)        
            self.painter.setPen(QPen(QColor(255, 255, 255), 12))
            self.painter.drawLine(self.lastPoint, self.endPoint)
            self.painter.end()
            self.update()
            self.lastPoint = self.endPoint

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()-self.offset
            self.update()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.label_8.setText(_translate("Form","predict="))
        self.loadimage2.setText(_translate("Form","loadimages"))
        self.showimages.setText(_translate("Form","5.1 show images"))
        self.showmodelresnet50.setText(_translate("Form","5.2 show model"))
        self.comparison.setText(_translate("Form","5.3 comparison"))
        self.inference.setText(_translate("Form","5.4 inference"))
        self.label_4.setText(_translate("Form", "4.MNIST Classifier Using VGG19"))
        self.reset.setText(_translate("MainWindow", "4. Reset"))
        self.predict.setText(_translate("MainWindow", "3. Predict"))
        self.histogramequalization.setText(_translate("MainWindow", "2.1 histogram equalization"))
        self.loadimage1.setText(_translate("MainWindow", "load image"))
        self.label_3.setText(_translate("MainWindow", "3.Morphology Operation"))
        self.label1.setText(_translate("MainWindow", "TextLabel"))
        self.closing.setText(_translate("MainWindow", "3.1 closing"))
        self.accuracyandloss.setText(_translate("MainWindow", "2. show accuracy and loss"))
        self.label_11.setText(_translate("MainWindow", "predict"))
        self.label_2.setText(_translate("MainWindow", "2.Histogram Equalization "))
        self.drawcontour.setText(_translate("MainWindow", "1.1.draw contour"))
        self.opening.setText(_translate("MainWindow", "3.2 opening"))
        self.showmodelvgg.setText(_translate("MainWindow", "1.show model structure"))
        self.label_6.setText(_translate("MainWindow", "There are __ coins in the image."))
        self.countcoins.setText(_translate("MainWindow", "1.2 countcoins"))
        self.label.setText(_translate("MainWindow", "1.Hough Circle Transform"))
        self.label_5.setText(_translate("MainWindow", "5.ResNet50"))

    def loadimage1_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "選擇圖像", "", "Image Files (*.jpg *.png *.bmp)")
        if file_path:
            self.image1 = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
            self.label1.setText(f"{os.path.relpath(file_path, os.getcwd())}")
        else:
            print("Failed to load or display image1")
        self.filepath=file_path
    def drawcontour_clicked(self):
        if self.image1 is not None:
            cv2.imshow('orignal',self.image1)
            black_image = np.zeros_like(self.image1)
            gray=cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            gua=cv2.GaussianBlur(gray,(3, 3),0)
            circles = cv2.HoughCircles(gua,cv2.HOUGH_GRADIENT,1.1,5,param1=50,param2=40,minRadius=20,maxRadius=40)
            circles=np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(self.image1,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.imshow('process',self.image1)
            for i in circles[0,:]:
                cv2.circle(black_image,(i[0],i[1]),2,(255,255,255),2)
            cv2.imshow('center',black_image)
    def countcoins_clicked(self):
        if self.image1 is not None:
            self.image1=cv2.imdecode(np.fromfile(self.filepath,dtype=np.uint8),-1)
            cv2.imshow('orignal',self.image1)
            gray=cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            gua=cv2.GaussianBlur(gray,(3, 3),0)
            circles = cv2.HoughCircles(gua,cv2.HOUGH_GRADIENT,1.1,5,param1=50,param2=40,minRadius=20,maxRadius=40)
            num=len(circles[0,:])
            self.label_6.setText(f"There are {num} coins in the image")
    def histogramequalization_clicked(self):
        if self.image1 is not None:
            gray_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            hist,bins = np.histogram(gray_image.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf,0)
            cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint8')
            cdf_image=cdf[gray_image]
            plt.subplot(2,3,1);plt.imshow( gray_image,cmap='gray')
            plt.title('Original Image')
            plt.subplot(2,3,2);plt.imshow( equalized_image,cmap='gray')
            plt.title('Equalized with opencv')
            plt.subplot(2,3,3);plt.imshow(cdf_image,cmap='gray')
            plt.title('Equalized with manual')
            plt.subplot(2, 3, 4), plt.hist(gray_image.flatten(), 256, [0, 256], color='b')
            plt.title('Original Histogram')
            plt.subplot(2, 3, 5), plt.hist(equalized_image.flatten(), 256, [0, 256], color='r')
            plt.title('Equalized Histogram(opencv)')
            plt.subplot(2,3,6);plt.hist(cdf_image.flatten(),256,[0,256],color='g')
            plt.title('Equalized Histogram(manual)')
            plt.show()
    def closing_clicked(self):
        if self.image1 is not None:
            cv2.imshow('orignal',self.image1)
            gray_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            _, binarized_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            padded_image = np.pad(binarized_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            ker=np.ones([3,3],dtype='int8')
            height,width=padded_image.shape[0],padded_image.shape[1]
            dilation_image=np.zeros((height-2,width-2),dtype=np.uint8)
            closing_image=np.zeros((height-2,width-2),dtype=np.uint8)
            for i in range(1,height-1):
                for j in range(1,width-1):
                    dilation_image[i-1, j-1] = np.max(padded_image[i-1:i+2, j-1:j+2] * ker) 
            padded_image = np.pad(dilation_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            for i in range(1,height-1):
                for j in range(1,width-1):
                    closing_image[i-1, j-1] = np.min(padded_image[i-1:i+2, j-1:j+2] * ker)
            cv2.imshow('closing',closing_image)   
    def opening_clicked(self):
        if self.image1 is not None:
            cv2.imshow('orignal',self.image1)
            gray_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            _, binarized_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            padded_image = np.pad(binarized_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            ker=np.ones([3,3],dtype='int8')
            height,width=padded_image.shape[0],padded_image.shape[1]
            erosion_image=np.zeros((height-2,width-2),dtype=np.uint8)
            opening_image=np.zeros((height-2,width-2),dtype=np.uint8)
            for i in range(1,height-1):
                for j in range(1,width-1):
                    erosion_image[i-1, j-1] = np.min(padded_image[i-1:i+2, j-1:j+2] * ker) 
            padded_image = np.pad(erosion_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            for i in range(1,height-1):
                for j in range(1,width-1):
                    opening_image[i-1, j-1] = np.max(padded_image[i-1:i+2, j-1:j+2] * ker)
            cv2.imshow('opening',opening_image)
    def showmodelvgg_clicked(self):
        model = models.vgg19_bn(num_classes=10)
        summary(model, (3, 32, 32))  
    def accuracyandloss_clicked(self):
        png= QPixmap()                   
        png.load('loss_and_accuracy.png')  
        png=png.scaled(self.pix.width(),self.pix.height())
        self.pix=png
        self.update()
    def predict_clicked(self):
        class VGG19Net(nn.Module):
            def __init__(self,num_classes=10):
                super(VGG19Net,self).__init__()
                self.model=nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
                self.classifier = nn.Sequential(
                    nn.Linear(7*7*512, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes)
                )
            def forward(self,x):
                x = self.model(x)
                x=self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
            
        class_name = ["0","1","2","3","4","5","6","7","8","9"]
        
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        self.pix.save(buffer, "PNG")
        bytes_io = io.BytesIO(buffer.data())
        pil_image = Image.open(bytes_io)
        pil_image = pil_image.convert("L")
        img=pil_image

        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                                    torchvision.transforms.ToTensor()])
        image=transform(img)
        model = VGG19Net()
        checkpoint = torch.load('best_model_vgg.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        image=torch.reshape(image,(1,1,32,32))
        with torch.no_grad():
            output=model(image)
            output_max=output.argmax(1)
            probabilities = F.softmax(output, dim=1)
            self.label_11.setText("predict={}".format(class_name[output_max.item()]))
            plt.figure(figsize=(8.5, 8.5))
            plt.bar(class_name, probabilities[0])
            plt.title("Probability of each class")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.xticks(rotation=45)
            plt.show()
    def reset_clicked(self):
        self.pix.fill(Qt.black)
        self.update()
    def loadimage2_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "選擇圖像", "inference_dataset", "Image Files (*.jpg *.png *.bmp)")
        if file_path:
            self.image1 = cv2.imread(file_path)
            qpixmap=QPixmap()
            qpixmap.load(file_path)
            re_qpixmap=qpixmap.scaled(224,224)
            self.label_7.setPixmap(re_qpixmap)
        else:
            print("Failed to load or display image1")
        self.re_qpixmap=re_qpixmap
    def showimages_clicked(self):
        class Mydata(Dataset):          
            def __init__(self,root_dir,label_dir):
                super().__init__()
                self.root_dir=root_dir
                self.label_dir=label_dir
                self.path=os.path.join(self.root_dir,self.label_dir)
                self.img_path=os.listdir(self.path)
            def __getitem__(self, idx):
                img_name=self.img_path[idx]
                img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
                img=Image.open(img_item_path)
                label=self.label_dir
                return img ,label
            def __len__(self):
                return len(self.img_path)            
        root_dir1="inference_dataset"
        label_dir1="Cat"
        catdata=Mydata(root_dir=root_dir1,label_dir=label_dir1)
        a=random.randint(0, 4)
        img_cat,lable_cat=catdata[a]
        root_dir2="inference_dataset"
        label_dir2="Dog"
        dogdata=Mydata(root_dir=root_dir2,label_dir=label_dir2)
        b=random.randint(0, 4)
        img_dog,lable_dog=dogdata[b]
        plt.figure(figsize=(12, 12))
        plt.subplot(1,2,1)
        plt.imshow(img_cat)
        plt.title(lable_cat)
        plt.subplot(1,2,2)
        plt.imshow(img_dog)
        plt.title(lable_dog)
        plt.show()
    def showmodelresnet50_clicked(self):
        resnet50_model = models.resnet50()
        num_ftrs = resnet50_model.fc.in_features
        resnet50_model.fc = nn.Linear(num_ftrs, 1)
        resnet50_model.fc= nn.Sigmoid()
        summary(resnet50_model, (3, 224, 224))
    def comparison_clicked(self):
        img=cv2.imread("comparison.png")
        cv2.imshow("comparison",img)
    def inference_clicked(self):
        class_name = ["cat","dog"]
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        self.re_qpixmap.save(buffer, "PNG")
        bytes_io = io.BytesIO(buffer.data())
        pil_image = Image.open(bytes_io)
        img=pil_image
        transform=torchvision.transforms.Compose([transforms.Resize((224)),
                                                  transforms.CenterCrop(224),
                                                  torchvision.transforms.ToTensor()])
        image=transform(img)
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        checkpoint = torch.load('best_model_resnet50_noeraser.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        image=torch.reshape(image,(1,3,224,224))
        with torch.no_grad():
            output=model(image)
            output_max=output.argmax(1)
            self.label_8.setText("pridict={}".format(class_name[output_max.item()]))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyDraw()
    window.show()
    sys.exit(app.exec_())
         