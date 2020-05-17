#%%
from explanation import Explanation
from generate_misclassification import GetMisclassification
from robustness import Robustness
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from alibi.datasets import fetch_imagenet
import numpy as np
import os, sys
try:
  import cPickle as pickle
except ModuleNotFoundError:
  import pickle

#%%
# Result dir
result_dir = os.getcwd() + '/results/imagenet'
if not os.path.exists(result_dir):
  os.makedirs(result_dir)

result_metadata_dir = result_dir + '/metadata/'
if not os.path.exists(result_metadata_dir):
  os.makedirs(result_metadata_dir)

#%%
# Setting up imagenet dataset
image_shape = (299, 299, 3)

categories = [
  'Persian cat',
  'Labrador retriever',
  'abacus', 'centipede',
  'digital watch',
  'balloon',
  'castle',
  'fountain',
  'library',
  'mask']

#%%
images = []
num_image_each_category = 1
for i, category in enumerate(categories):
  data, _  = fetch_imagenet(
    category,
    nb_images=num_image_each_category,
    target_size=image_shape[:2],
    seed=2,
    return_X_y=True)
  
  for n in range(num_image_each_category):
    images.append(data[n])

images = np.asarray(images)

# save images to metadata
images_pkl_filename = result_metadata_dir + 'data.pkl'

with open(images_pkl_filename, 'wb') as output:
  pickle.dump(images, output, pickle.HIGHEST_PROTOCOL)

#%%
#########
from keras.preprocessing import image
images = []

img_path = '/Users/rohitsingh/Desktop/Thesis/lime/lime/doc/notebooks/data/cat.jpg'
images.append(image.load_img(img_path, target_size=(299, 299)))

img_path = '/Users/rohitsingh/Desktop/Thesis/lime/lime/doc/notebooks/data/cat_mouse.jpg'
images.append(image.load_img(img_path, target_size=(299, 299)))

images[0] = np.asarray(images[0])
images[1] = np.asarray(images[1])

images = np.asarray(images)

# save images to metadata
images_pkl_filename = result_metadata_dir + 'data.pkl'

with open(images_pkl_filename, 'wb') as output:
  pickle.dump(images, output, pickle.HIGHEST_PROTOCOL)

#%%

with open('/Users/rohitsingh/Desktop/project/anchor/results/imagenet/metadata/data.pkl', 'rb') as input:
  images = pickle.load(input)


methods = [
  'Anchor',
  'LIME',
  'saliency',
  'grad*input',
  'intgrad',
  'elrp',
  'deeplift',
  'occlusion',
  'shapley_sampling']

for method in methods:
  # Get explanation i.e anchors
  exp = Explanation(exp_method=method, exp_model="InceptionV3")
  res = exp.GetExplanation(exp_type="image", exp_data=images)
  # save explanation
  exp_res_pkl_filename = result_metadata_dir + method + '_explanation.pkl'

  with open(exp_res_pkl_filename, 'wb') as output:
    pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)

  # Get misclassification on anchors and anchor's complemets
  misclass_obj = GetMisclassification(exp_metadata=res, exp_data=images, exp_dataset="imagenet")

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



