#%%
from explanation import Explanation
from generate_misclassification import GetMisclassification
from robustness import Robustness
from skimage.color import gray2rgb, rgb2gray, label2rgb
from skimage.util import montage as montage2d
from sklearn.datasets import fetch_openml
import numpy as np
import os, sys
try:
  import cPickle as pickle
except ModuleNotFoundError:
  import pickle

#%%
# Result dir
result_dir = os.getcwd() + '/results/mnist'
if not os.path.exists(result_dir):
  os.makedirs(result_dir)

result_metadata_dir = result_dir + '/metadata/'
if not os.path.exists(result_metadata_dir):
  os.makedirs(result_metadata_dir)

#%%
# Setting up imagenet dataset
image_shape = (28, 28, 3)

#%%
# Create a custom model
from sample_nn import GetnnModel

model = GetnnModel(dataset_name="MNIST")

# save model to metadata
model_pkl_filename = result_metadata_dir + 'model.pkl'

with open(model_pkl_filename, 'wb') as output:
  pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

#%%
# Load the test data
mnist = fetch_openml('mnist_784')
# make each image color so lime_image works correctly
X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))],0)
y_vec = mnist.target.astype(np.uint8)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                    train_size=0.55)

images = x_test[:5000]

# save images to metadata
images_pkl_filename = result_metadata_dir + 'data.pkl'

with open(images_pkl_filename, 'wb') as output:
  pickle.dump(images, output, pickle.HIGHEST_PROTOCOL)



#%%
# explanation methods
methods = ['saliency']

# methods = [
#   'Anchor',
#   'LIME',
#   'saliency',
#   'grad*input',
#   'intgrad',
#   'elrp',
#   'deeplift',
#   'occlusion',
#   'shapley_sampling']

for method in methods:
  # Get explanation i.e anchors
  exp = Explanation(exp_method=method, predict_fn=model.predict, exp_model=model)
  res = exp.GetExplanation(exp_type="image", exp_data=images[:50], exp_model=model)
  exp_res_pkl_filename = result_metadata_dir + method + '_explanation.pkl'

  with open(exp_res_pkl_filename, 'wb') as output:
    pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)

  # Get misclassification on anchors and anchor's complemets
  misclass_obj = GetMisclassification(exp_metadata=res, exp_data=images[:50], predict_fn=model.predict)

  metadata = misclass_obj.AddNoiseForMisclassification()
  # save misclassified data
  misclassification_res_pkl_filename = result_metadata_dir + method + '_misclassification.pkl'

  with open(misclassification_res_pkl_filename, 'wb') as output:
    pickle.dump(metadata, output, pickle.HIGHEST_PROTOCOL)
  
  # get visual data
  num = misclass_obj.misclassified_data['F']
  filename = result_dir + '/' + method + '_result.pdf'

  misclass_obj.GiveVisual(num=num, filename=filename)

  # Get robustness score
  robustness = Robustness([metadata])

  # save robustness data
  robustness_res_pkl_filename = result_metadata_dir + method + '_robustness.pkl'

  with open(robustness_res_pkl_filename, 'wb') as output:
    pickle.dump(robustness.robustness_data, output, pickle.HIGHEST_PROTOCOL)

  # save the result in txt file
  filename = result_dir + '/' + method + '_robustness.txt'

  robustness.MakeReport(filename)




# %%
