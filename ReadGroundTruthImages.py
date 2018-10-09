import SimpleITK as sitk
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class Patient: 
    def __init__(self, id, mamoImageCC,mamoImageMLO,side):
        self.id = id 
        self.mamoImageCC = mamoImageCC
        self.mamoImageMLO = mamoImageMLO
        self.side = side
        
    def getId (self):
        return self.id
    
    def getMamoImageCC(self):
        if (np.all(self.mamoImageCC != None)):
            return self.mamoImageCC
        else:
            print('Has not found CC Image')
            return None
        
    def getMamoImageMLO(self):
        if (np.all(self.mamoImageMLO != None)):
            return self.mamoImageMLO
        else:
            print('Has not found MLO Image')
            return None
        
    def getSide(self):
        if(self.side != None):
            return self.side
    
    def setMamoImageCC(self,mamoImageCC):
        self.mamoImageCC = mamoImageCC
    
    def setMamoImageMLO(self, mamoImageMLO):
        self.mamoImageMLO = mamoImageMLO
    
    def setSide(self, side):
        self.side = side
        
    
    
class ListOfPatients:
    listofPatients = []
    
#    def __init__(self):
        
    def addAPatient(self, pa):
        self.listofPatients.append(pa)
        
    def getPatientbyId(self, ID):
        for p in self.listofPatients:
            if p.getId() == ID: 
                print('found!')
                return p
        return None 
#        pa = (p for p in self.listofPatients if p.getId() == ID )
#        print(pa.getId())
#        return pa
    
    def getAllPatientsbySide(self, side):
        print("here we go with side")
#        pas = (p for p in self.listofPatients if p.side == side)
#        return pas
       

def load_itk(filename):
    
    #Read image using simpleITK
    itkimage = sitk.ReadImage(filename)
    
    #Convert the image into numpy array and then shuffle the dimentions to get axis in the order z, y, x
    mamograph = sitk.GetArrayFromImage(itkimage)
    return itkimage, mamograph


def read_images():
    PATH = os.getcwd()
    Segmented_directory = PATH + '/Current_segmented'
    
    #list of all patients folders
    Pa_dir_list = os.listdir(Segmented_directory)
    LOP = ListOfPatients()
    for PatientsID in Pa_dir_list:
    
        patient_directory = Segmented_directory+'/'+PatientsID
        
        #list of images per patient
        im_dir_list = os.listdir(patient_directory)
        newP = Patient(PatientsID,None,None,None)
        for Images in im_dir_list:
            
            #filter images that shows CC view
            if(Images[1:] == 'CC.mhd'):
                print(PatientsID)
                im, nparray = load_itk(patient_directory+'/'+Images)
                nparray = np.squeeze(nparray,0)
                res = cv2.resize(nparray, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
                newP.setMamoImageCC(res)
                newP.setSide(Images[0])
                
                #flip the images if it belongs to right breast
                if(Images[0] =='R'):
                    res = cv2.flip(res,1)
           
            if(Images[1:] == 'MLO.mhd'):
                im, nparray = load_itk(patient_directory+'/'+Images)
                nparray = np.squeeze(nparray,0)
                res = cv2.resize(nparray, dsize=(224,224), interpolation = cv2.INTER_CUBIC)
                newP.setMamoImageMLO(res)
                newP.setSide(Images[0])
         
            LOP.addAPatient(newP)   
        
    return LOP

# can call read_images to get all the patients information


