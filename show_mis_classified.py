import matplotlib.pyplot as plt

#rows : rows in image block grid
#columns: column in image block grid
#type : misclassified/ParticularClass/Random/MisclassifiedClass
#class : class of misclassified image
#y_pred

def showAndPlotImage(rows, columns, type, class, y_pred, y_test,x_test):
  if(type == "misclassified"):
    # Generate a rows x cols sized image grid 
    f, axarr = plt.subplots(rows,cols)

    i = 0
    count = 0
    while ((count < rows*cols) and (i < x_test.shape[0])):
       if(predictedDigits[i] != y_test[i]):
            axarr[count // rows][count % cols].imshow(x_test[i])
            axarr[count // rows][count % cols].set_title('Predicted:{} / Actual:{}'.format(predictedDigits[i],y_test[i]),fontsize=12)
            axarr[count // rows][count % cols].axis('off')
            count += 1
       
       i +=  1
                 
    f.subplots_adjust(hspace=0.2)    
    f.suptitle('List of Missclassified images', fontsize=25)
    f.set_size_inches(15,15)
    

  
