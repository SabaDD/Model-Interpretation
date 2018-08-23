import numpy as np
import xlrd
import matplotlib.pyplot as plt
import scipy
from keras.preprocessing import image
from PIL import Image
import ast
import seaborn as sns

sns.set()
#img_sample = image.load_img('Cancer_Sample.jpg', target_size= (224,224))
#plt.figure
#plt.imshow(img_sample)
img = np.zeros(shape = (224,224,4))
plt.imshow(img)
wb = xlrd.open_workbook('Results_small_sliding_window_56_6_not_Shuffle_not_Mean.xlsx')
sheet = wb.sheet_by_index(0)
datalist = np.zeros(shape=(16,10))
for j in range (0,sheet.ncols):
    for i in range (1,sheet.nrows):
        k = float(sheet.cell_value(i,j))
        if(k < 0.5):
            datalist[i-1,j] = 1-k
        else:
            datalist[i-1,j] = k

averageSW = []            
for i in range (0, sheet.nrows-1):
    averageSW.append(np.average(datalist[i,:]))


newTable = np.zeros(shape=(4,4))
p = 0
for k in range (0,10):
    for i in range (0,4):
        for j in range (0,4):
    
            newTable[j,i] = datalist[i+(i*3)+j,k]
    plt.figure(k)       
    ax = sns.heatmap(newTable, vmin=0.5, vmax=0.8)

    fig = ax.get_figure()
    fig.savefig('result_6/'+str(k)+'.png')
#print('hh1')
#a = 0
#b = 255  
##data_norm_by_std = [number/scipy.std(datalist) for number in datalist]
#data_norm_to_a_b = [(number - a)/(b - a) for number in datalist]
#print('hh2')
#for i in range (0,sheet.nrows-1):
#    print('i ={}'.format(i))    
#    left,upper = divmod(i,112)
#    for x in range(left, left+112):
#        for y in range (upper, upper+112):
#            if img[x,y,3] < datalist[i]:
#                img[x,y,3] = datalist[i]
#                if 0.4525< datalist[i] < 0.5059:
#                    img[x,y,0] = 0
#                    img[x,y,1] = 0
#                    img[x,y,2] = 400*data_norm_to_a_b[i]
#                elif 0.5059< datalist[i] < 0.6:
#                    img[x,y,0] = 0
#                    img[x,y,1] = 400*data_norm_to_a_b[i]
#                    img[x,y,2] = 0
#                elif 0.6< datalist[i] < 0.62:
#                    img[x,y,0] = 400*data_norm_to_a_b[i]
#                    img[x,y,1] = 0
#                    img[x,y,2] = 0

##img_resize =  Image.fromarray(img[:,:,0:3], 'RGB')
##plt.figure
##plt.imshow(img_resize)
##combined_image = (img_sample + img[:,:,0:3]) / 2
#plt.figure
#plt.imshow(img)
#img_sample1 = image.load_img('Cancer_Sample2.jpg', target_size= (224,224))
#img_sample2 = image.load_img('Cancer_Sample3.jpg', target_size= (224,224)) 
#img_map = image.load_img('chart3-0.56.jpg', target_size= (224,224))    
#               
#image5 = img_sample1.convert("RGBA")
#image6 = img_sample2.convert("RGBA")
#image7 = img_map.convert("RGBA")
#
## Display the images
##image5.show()
##image6.show()
##image7.show()
#
## alpha-blend the images with varying values of alpha
#alphaBlended1 = Image.blend(image7, image5, alpha=.5)
#alphaBlended2 = Image.blend(image7, image6, alpha=.5)
#
## Display the alpha-blended images
#alphaBlended1.show()
#alphaBlended2.show() 
