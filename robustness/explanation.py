import numpy as np
from alibi.explainers import AnchorImage
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from skimage.color import gray2rgb, rgb2gray, label2rgb
from skimage.util import montage as montage2d
from keras import backend as K
import timeit

import torch
from torchvision import models, transforms

class Explanation:

  def __init__(self, exp_method, exp_model=None, predict_fn=None, dataset_name=None):
    '''
    exp_method
      Type of explantion method:
        Anchor, LIME, saliency, grad*input,
        intgrad, elrp, deeplift, occlusion,
        shapley_sampling
    exp_model
      Model to explain: Inception, RF, SVM, sample nn
    '''
    # K.clear_session()
    self.exp_method = exp_method
    self.exp_model = exp_model
    self.exp_data = None
    self.dataset_name = dataset_name
    self.exp_type = None
    self.predict_fn = predict_fn
    self.explanations_labels = {
      'method': self.exp_method,
      # True lables list [data point][ith prediction][(class name, pred acc)]
      'labels': [],
      # Anchor points [data point][anchor points (pixels location in case of img)]
      'explanation_anchor': None,
      # Points that impact negative to prediction [data point][pts]
      # Gradient based method give this kind of representations
      'explanation_anti_anchor': None
    }

  def GetExplanation(self, exp_type, exp_data, dataset_name=None, exp_model=None):
    '''
    data
      [N,H,W,C]
        N: Number of images (0 for only one image)
        H: Height
        W: width
        C: Channels
    retrun
      self.explanations_labels
    '''
    if self.dataset_name is None:
      self.dataset_name = dataset_name
    if self.exp_model is None:
      self.exp_model = exp_model
    
    deep_explain_methods = [
      "saliency",
      "grad*input",
      "intgrad",
      "elrp",
      "deeplift",
      "occlusion",
      "shapley_sampling"]

    self.exp_data = exp_data
    self.exp_type = exp_type
    # this for image
    N = 0
    if exp_type == "image":
      N, _, _, _ = self.exp_data.shape
    self.explanations_labels['labels'] = []
    self.explanations_labels['explanation_anchor'] = []
    self.explanations_labels['explanation_anti_anchor'] = []

    if self.exp_type == "image":
      if self.exp_method == "Anchor":
        self.AnchorImage()
        return self.explanations_labels
      
      if self.exp_method == "LIME":
        self.LimeImage()
        return self.explanations_labels
      
      if self.exp_method in deep_explain_methods:
        if self.dataset_name is None and self.exp_model is None:
          raise Exception("Error: provide model or select dataset for this method from MNIST, "\
            "olivetti_faces, imagenet.")

        self.DeepExplain()
        return self.explanations_labels
      
      methods = 'Anchor, LIME, saliency, grad*input, intgrad, elrp, '\
      'deeplift, occlusion, shapley_sampling'

      raise Exception("Error: method {} not found. Please use one of "\
        "these -\n {}".format(self.exp_method, methods))
      

  
  def AnchorImage(self):
    '''
    Give Anchor method explanation on given data
    for given model
    '''
    image_shape = self.exp_data.shape[1:]
    if self.predict_fn == None:

      if self.exp_model == "InceptionV3":
        K.clear_session()
        model = InceptionV3(weights='imagenet')
        image_shape = (299, 299, 3)

        self.exp_data = preprocess_input(self.exp_data)

        preds = model.predict(self.exp_data)
        labels = decode_predictions(preds, top=1000)

        N, _, _, _ = self.exp_data.shape
        for i in range(N):
          tmp = []
          for j in range(1000):
            tmp.append(labels[i][j][1:])
          self.explanations_labels['labels'].append(tmp)

        self.predict_fn = lambda x: model.predict(x)
      
      # else:
      #   self.predict_fn = model.predict
    
    else:
        prob = self.predict_fn(self.exp_data)
        N, _, _, _ = self.exp_data.shape
        for i in range(N):
          tmp = []
          for j in range(len(prob[0])):
            tmp.append((str(j), prob[i][j]))
          self.explanations_labels['labels'].append(tmp)

    if self.predict_fn == None:  
      raise Exception("Error no prediction function or valid model is given.")
    
    ###

    beam_size=1
    threshold=.95
    segmentation_fn = 'slic'
    kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5}
    N, row, col, ch = self.exp_data.shape

    if row <= 128 or col <= 128:
      from skimage import segmentation

      if row == 28 and col == 28:
        # MNIST
        beam_size=2
        threshold=.98
        segmentation_fn = segment = segmentation.felzenszwalb
        kwargs = {'scale': 50, 'min_size': 50, 'sigma': .1}
      elif row == 64 and col == 64:
        # olivetti_faces
        beam_size=2
        threshold=.98
    ###
    
    explainer = AnchorImage(self.predict_fn, image_shape, segmentation_fn=segmentation_fn,
                            segmentation_kwargs=kwargs, images_background=None)
   

    print('Generating explanation for method {}'.format(self.exp_method))
    for n in range(N):
      # self.explanations_labels['explanation_anchor'][n] = []
      anchor_points = []
      # self.explanations_labels['labels'] = self.predict_fn(self.exp_data[n])
      start_time = timeit.default_timer()
      explanation = explainer.explain(self.exp_data[n], threshold=threshold, p_sample=.5, tau=0.25, beam_size=beam_size)
      print('  Data Point {} explanation took {:.2f} sec'.format(n, timeit.default_timer() - start_time))

      segments_array = explanation.data['segments']
      superpixels_in_anchor = explanation.data['raw']['feature']
      for i in range(row) :
        for j in  range(col):
          if segments_array[i][j] in superpixels_in_anchor:
            anchor_points.append([i, j])
      self.explanations_labels['explanation_anchor'].append(anchor_points)

  def LimeImage(self):
    '''
    Give LIME method explanation on given data
    for given model
    '''
    image_shape = self.exp_data.shape[1:]
    if self.predict_fn == None:

      if self.exp_model == "InceptionV3":
        K.clear_session()
        model = InceptionV3(weights='imagenet')
        image_shape = (299, 299, 3)

        self.exp_data = preprocess_input(self.exp_data)

        preds = model.predict(self.exp_data)
        labels = decode_predictions(preds, top=1000)

        N, _, _, _ = self.exp_data.shape
        for i in range(N):
          tmp = []
          for j in range(1000):
            tmp.append(labels[i][j][1:])
          self.explanations_labels['labels'].append(tmp)

        self.predict_fn = lambda x: model.predict(x)
      

    else:      
        
        prob = self.predict_fn(self.exp_data)
        N, _, _, _ = self.exp_data.shape
        for i in range(N):
          tmp = []
          for j in range(len(prob[0])):
            tmp.append((str(j), prob[i][j]))
          self.explanations_labels['labels'].append(tmp)

    if self.predict_fn == None:  
      raise Exception("Error no prediction function or valid model is given.")

    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    explainer = lime_image.LimeImageExplainer(verbose = False)
    # segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    N, row, col, ch = self.exp_data.shape
    num_samples = 1000
    segmenter = None

    if row <= 128 or col <= 128:
      num_samples = 10000
      if row == 28 and col == 28:
        # MNIST
        segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
      elif row == 64 and col == 64:
        # olivetti_faces
        segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)
    
    print('Generating explanation for method {}'.format(self.exp_method))
    for n in range(N):
      # self.explanations_labels['explanation_anchor'][n] = []
      anchor_points = []
      start_time = timeit.default_timer()
      explanation = explainer.explain_instance(
        self.exp_data[n],
        # np.array(pill_transf((transforms.ToPILImage()(self.exp_data[n])))), ### TESTING
        classifier_fn = self.predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmenter)
      print('  \nData Point {} explanation took {:.2f} sec'.format(n, timeit.default_timer() - start_time))

      temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=True)
      
      for i in range(row) :
        for j in  range(col):
          if np.count_nonzero(temp[i][j]) > 0:
            anchor_points.append([i, j])

      self.explanations_labels['explanation_anchor'].append(anchor_points)
  
  def DeepExplain(self):
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Flatten, Activation
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    from imageio import imread
    import tensorflow as tf
    from tensorflow.contrib.slim.nets import inception

    slim = tf.contrib.slim

    # Import DeepExplain
    from deepexplain.tensorflow import DeepExplain


    if self.exp_model == "InceptionV3" or self.dataset_name == "imagenet":
      # Assigning labels
      K.clear_session()
      self.exp_model = InceptionV3(weights='imagenet')
      image_shape = (299, 299, 3)

      self.exp_data = preprocess_input(self.exp_data)

      preds = self.exp_model.predict(self.exp_data)
      labels = decode_predictions(preds, top=1000)

      N, row, col, ch = self.exp_data.shape
      for i in range(N):
        tmp = []
        for j in range(1000):
          tmp.append(labels[i][j][1:])
        self.explanations_labels['labels'].append(tmp)

      ## TODO
      import sys, os
      sys.path.append(os.getcwd())

      # Load Inception V3 model from Tensorflow Slim, restore section
      # from checkpoint and run the classifier on the input data
      num_classes = 1001

      # Select the model here. Use adv_inception_v3 to use the weights of
      # an adversarially trained Inception V3. Explanations will be more sparse.

      checkpoint = 'data/models/inception_v3.ckpt'
      # checkpoint = 'data/models/adv_inception_v3.ckpt'

      tf.reset_default_graph()
      sess = tf.Session()

      # Since we will explain it, the model has to be wrapped in a DeepExplain 
      # context
      with DeepExplain(session=sess, graph=sess.graph) as de:
          X = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

          with slim.arg_scope(inception.inception_v3_arg_scope()):
              tmp, end_points = inception.inception_v3(
                X,
                num_classes=num_classes,
                is_training=False)

          logits = end_points['Logits']
          yi = tf.argmax(logits, 1)

          saver = tf.train.Saver(slim.get_model_variables())
          saver.restore(sess, checkpoint)

          # filenames, xs = load_images()
          # labels = sess.run(yi, feed_dict={X: xs})
          # print (filenames, labels)

      # Compute attributions for the classified images
      # Every DeepExplain method must be called in a DeepExplain context.
      # In this case, we use two different contexts to create the model and to 
      # run the explanation methods. This works as long as the same session is
      #  provided.
      print('Generating explanation....')
      start_time = timeit.default_timer()
     
      with DeepExplain(session=sess) as de:
        if self.exp_method == "occlusion":
          attributions_exp = de.explain(
            'occlusion',
            tf.reduce_max(logits, 1),
            X,
            self.exp_data,
            window_shape=(15,15,3))
        elif self.exp_method == "shapley_sampling":
          attributions_exp = de.explain(
            'shapley_sampling',
            tf.reduce_max(logits, 1),
            X,
            self.exp_data,
            samples=100)
        else:  
          attributions_exp = de.explain(
            self.exp_method,
            tf.reduce_max(logits, 1),
            X,
            self.exp_data)
            # attributions = {
              # Gradient-based
              # NOTE: reduce_max is used to select the output unit for the 
              # class predicted by the classifier
              # For an example of how to use the ground-truth labels instead,
              # see mnist_cnn_keras notebook
              # 'Saliency maps':        de.explain('saliency', 
              #   tf.reduce_max(logits, 1), X, xs),
              # 'Gradient * Input':     de.explain('grad*input',
              #   tf.reduce_max(logits, 1), X, xs),
              # 'Integrated Gradients': de.explain('intgrad',
              #   tf.reduce_max(logits, 1), X, xs),
              # 'Epsilon-LRP':          de.explain('elrp',
              #   tf.reduce_max(logits, 1), X, xs),
              # 'DeepLIFT (Rescale)':   de.explain('deeplift',
              #   tf.reduce_max(logits, 1), X, xs),
              # Perturbation-based (comment out to evaluate, but this will take
              # a while!)
              # 'Occlusion [15x15]':    de.explain('occlusion', 
              #   tf.reduce_max(logits, 1), X, xs, window_shape=(15,15,3), step=4)
          # }
        print('\n\n    Finished took {:.2f}'.format( timeit.default_timer() - start_time))

      

      # Setting anchor pixels
      for i in range(N):
        pos_anchor = []
        neg_anchor = []
        
        mean_ = np.mean(attributions_exp[i])
        max_ = np.max(attributions_exp[i])
        tmp = mean_ + (max_ - mean_) / 10

        for r in range(row):
          for c in range(col):
            if ch == 1:
              if attributions_exp[i][r][c] > 0.05:
                pos_anchor.append([r, c])
              if attributions_exp[i][r][c] < -0.05:
                neg_anchor.append([r, c]) 
            else:
              count = 0
              for k in range(ch):
                if attributions_exp[i][r][c][k] > tmp:
                  count += 1
              if count == 0:
                pos_anchor.append([r, c])
        
        self.explanations_labels['explanation_anchor'].append(pos_anchor)
        self.explanations_labels['explanation_anti_anchor'].append(neg_anchor)

    
    else:
      import sys, os
      sys.path.append(os.getcwd())
      from sample_nn import GetnnModel

      if self.exp_model is None:
        print('Building model for given dataset...')
        self.exp_model = GetnnModel(self.dataset_name)
        print('    Finished.')
      # DeepExplain


      print('Generating explanation....')
      start_time = timeit.default_timer()
      with DeepExplain(session=K.get_session()) as de:  # <- init DeepExplain
        # Need to reconstruct the graph in DeepExplain context, using the 
        # same weights.
        # With Keras this is very easy:
        # 1. Get the input tensor to the original model
        input_tensor = self.exp_model.layers[0].input
        
        # 2. We now target the output of the last dense layer (pre-softmax)
        # To do so, create a new model sharing the same layers untill the last
        # dense (index -2)
        fModel = Model(inputs=input_tensor, outputs = self.exp_model.layers[-2].output)
        target_tensor = fModel(input_tensor)
        
        xs = self.exp_data
        ys = self.exp_model.predict(xs)

        # Setting explanations_labels['labels']
        N, row, col, ch = self.exp_data.shape
        for i in range(N):
          tmp = []
          for j in range(len(ys[0])):
            tmp.append((str(j), ys[i][j]))
          self.explanations_labels['labels'].append(tmp)
        
        attributions_exp = de.explain(
          self.exp_method,
          target_tensor,
          input_tensor,
          xs,
          ys=ys)
        
        # Setting anchor pixels
        for i in range(N):
          pos_anchor = []
          neg_anchor = []
          # when channels = 3

          for r in range(row):
            for c in range(col):
              if ch == 1:
                if attributions_exp[i][r][c] > 0.05:
                  pos_anchor.append([r, c])
                if attributions_exp[i][r][c] < -0.05:
                  neg_anchor.append([r, c]) 
              else:
                if np.mean(attributions_exp[i][r][c]) > 0.04:
                  pos_anchor.append([r, c])
          
          self.explanations_labels['explanation_anchor'].append(pos_anchor)
          self.explanations_labels['explanation_anti_anchor'].append(neg_anchor)
      
      print('\n\n    Finished. took {:.2f}'.format( timeit.default_timer() - start_time))





  
