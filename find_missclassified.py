from keras.models import load_model
from keras.preprocessing.image import save_img
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def find_misclassified_images (model_path, test_iterator, test_label): 
#Load the best saved model and create predictions on the test set

  mnist_model = load_model(model_path)

  probabilities = mnist_model.predict_generator(test_iterator,steps=len(test_iterator))

  #print(probabilities[0])
  y_pred = np.argmax(probabilities, axis=1)


  #See which we predicted correctly and which not

  correct_indices = np.nonzero(y_pred == test_label)[0] 
  incorrect_indices = np.nonzero(y_pred != test_label)[0]
  print(len(correct_indices)," classified correctly") 
  print(len(incorrect_indices)," classified incorrectly")

  return incorrect_indices

def print_save_misclassified_images(save_flag,save_path,incorrect_indices,number):
  
  for i, incorrect in enumerate(incorrect_indices[:number]):
    
  if save_flag == 1 :
    filepath2="{}{}.{}".format(save_path,i,".jpg")
  
  #print(filepath2)
  
  save_img(filepath2, X_test[incorrect])

  plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

  plt.title( "Predicted {}, Truth: {}".format(y_pred[incorrect], y_test[incorrect]))

  plt.xticks([])

  plt.yticks([])

  plt.show()
