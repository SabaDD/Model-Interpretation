import numpy as np
from sklearn.metrics import roc_curve, auc
from image_preprocessing import change_data



def model_prediction(custom_resnet_model, X_test, y_test, left,right,upper,lower):
    
    roc_auc_cv = []
    new_list = change_data(X_test,LEFT = left,RIGHT = right,UPPER = upper,LOWER = lower)
    new_list = np.array(new_list)
#    plt.figure
#    plt.imshow((new_list[0] * 255).astype(np.uint8))
#    print('this is the new list: ============> '+ str(new_list[0]))
    # Plot ROC_AUC curve
    
    y_score = custom_resnet_model.predict(new_list)
    pred = y_score[:, 1]  # only positive cases
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test[:,1], pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
#    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
#    plt.legend(loc='lower right')
#    plt.xlabel('False positive rate')
#    plt.ylabel('True positive rate')
    #plt.show()
    roc_auc_cv.append(roc_auc)
#    plt.savefig('Plots/Auc'+ str(ii)+'-'+str(k)+'.jpg')
#    plt.close()
    
    
#    print("Average AUC= %.2f "%np.mean(roc_auc_cv))
    
    return np.mean(roc_auc_cv)
