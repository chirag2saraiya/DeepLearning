from keras.models import load_model
from keras.preprocessing.image import save_img
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#Give incorrect indices 
#model_path:        saved model path
#test_iterator :    test iterator
#test_label :       test label array

"""API to find imdices of misclassified images """

def find_misclassified_images (model, test_iterator, test_label): 

  probabilities = model.predict_generator(test_iterator,steps=len(test_iterator))

  #print(probabilities[0])
  y_pred = np.argmax(probabilities, axis=1)


  #See which we predicted correctly and which not

  correct_indices = np.nonzero(y_pred == test_label)[0] 
  incorrect_indices = np.nonzero(y_pred != test_label)[0]
  print(len(correct_indices)," classified correctly") 
  print(len(incorrect_indices)," classified incorrectly")

  return incorrect_indices,y_pred


"""API to Save and Print misclassified images """

def print_save_misclassified_images(model, test_iterator, test_label, x_test, number_of_image, grid_width, grid_height, save_path = "",save_flag=0):
    
    incorrect_indices,y_pred = find_misclassified_images(model, test_iterator, test_label)
    
    if save_flag == 1 :
      filepath2="{}{}.{}".format(save_path,i,".jpg")
      save_img(filepath2, X_test[incorrect])
      
    f, ax = plt.subplots(grid_width, grid_height)
    f.set_size_inches(12, 12)
    
    img_idx = 0
    for i in range(0, grid_width):
        for j in range(0, grid_height):
            ax[i][j].axis('off')
            ax[i][j].set_title("Predicted {}, Truth: {}".format(y_pred[incorrect_indices[img_idx]], y_test[incorrect_indices[img_idx]]))
            ax[i][j].imshow(image_data[img_idx]),(im_scale_x,im_scale_y)))
            img_idx += 1
            
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.25)55)
       

def show_image_grid(image_data=[] , number_of_image, class_label = [], grid_width, grid_height, im_scale_x, im_scale_y):
  f, ax = plt.subplots(grid_width, grid_height)
  f.set_size_inches(12, 12)
  
  img_idx = 0
  for i in range(0, grid_width):
    for j in range(0, grid_height):
      ax[i][j].axis('off')
      ax[i][j].set_title("Label {}".format(y_pred[incorrect], y_test[incorrect]))
      indx = np.random.randint(image_data.shape[0])
      ax[i][j].imshow(image_data[indx]),(im_scale_x,im_scale_y)))
      img_idx = img_idx+1
      if(img_idx>=number_of_image):
        break
        
            
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.25)55)
    
