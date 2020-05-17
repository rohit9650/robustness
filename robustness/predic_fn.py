import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, label2rgb
from skimage.util import montage as montage2d
import timeit

class PredictionFunction:

  def __init__(self, dataset_name, model_name):
    self.dataset_name = dataset_name
    self.model_name = model_name
    self.predict_fn = None
  
  def GetPredictinFunction(self):
    X_train, X_test, y_train, y_test = None, None, None, None

    if self.dataset_name == "olivetti_faces":
      from sklearn.datasets import fetch_olivetti_faces
      faces = fetch_olivetti_faces()
      # make each image color so lime_image works correctly
      X_vec = np.stack([gray2rgb(iimg) for iimg in faces.data.reshape((-1, 64, 64))],0)
      y_vec = faces.target.astype(np.uint8)

      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                          train_size=0.70)

    if self.dataset_name == "MNIST":
      from sklearn.datasets import fetch_openml
      mnist = fetch_openml('mnist_784')
      # make each image color so lime_image works correctly
      X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))],0)
      y_vec = mnist.target.astype(np.uint8)

      from sklearn.model_selection import train_test_split

      X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec,
                                                          train_size=0.55)


    if (type(X_train) == type(None)) or (type(y_train) == type(None)):
      raise Exception("Error no valid dataset provided")

    if self.model_name == "rf":
      self.RandomForest(X_train, X_test, y_train, y_test)

      return self.predict_fn

  
  def RandomForest(self, X_train, X_test, y_train, y_test):
      
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import Normalizer
    from sklearn.decomposition import PCA

    class PipeStep(object):
        """
        Wrapper for turning functions into pipeline transforms (no-fitting)
        """
        def __init__(self, step_func):
            self._step_func=step_func
        def fit(self,*args):
            return self
        def transform(self,X):
            return self._step_func(X)

    makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
    flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

    simple_rf_pipeline = Pipeline([
        ('Make Gray', makegray_step),
        ('Flatten Image', flatten_step),
        ('Normalize', Normalizer()),
        ('PCA', PCA(25)),
        ('XGBoost', GradientBoostingClassifier())
                                  ])
    print("Training random forest....")
    start_time = timeit.default_timer()
    simple_rf_pipeline.fit(X_train, y_train)
    print('   Training took {:.2f} sec'.format(timeit.default_timer() - start_time))

    pipe_pred_test = simple_rf_pipeline.predict(X_test)
    pipe_pred_prop = simple_rf_pipeline.predict_proba(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy = {:.2f}'.format(accuracy_score(y_true=y_test, y_pred = pipe_pred_test) * 100))

    self.predict_fn = simple_rf_pipeline.predict_proba