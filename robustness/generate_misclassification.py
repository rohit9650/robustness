import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import timeit
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

class GetMisclassification:
  def __init__(
    self,
    exp_metadata,
    exp_data,
    predict_fn= None,
    exp_dataset=None):
    '''
    class to generate misclassification via adding noise to anchor and 
      complement of anchor.
    exp_metadata
      {
        'method': method name
      # True lables list [data point][ith prediction][(class name, pred acc)]
        'labels': [],
      # Anchor points [data point][anchor points (pixels location in case of img)]
        'explanation_anchor': None,
      # Points that impact negative to prediction [data point][pts]
      # Gradient based method give this kind of representations
        'explanation_anti_anchor': None
      }
    exp_data
      data on which we will add noise (ex: batch of images)
    predict_fn
      prediction function to predict the output

    return
      misclassified data 
        {   
          'anchor'
            {
              'p': list of bin value 
              'data': The misclassfied data
              'original_data_idx' : idx of original data
              'labels' : list of misclassified example labels 
                         (when noise is added to anchor)
              'original_labels' :
            }
          'anchor_comp'
            {
              'p': list of bin value 
              'data': The misclassfied data
              'original_data_idx :  idx of original data
              'labels' : list of misclassified example labels
                         (when noise is added to anchor's complement)
              'original_labels' :
            }
            'F': # misclassified examples
        }
    '''
    self.exp_metadata = exp_metadata
    self.exp_data = exp_data
    self.exp_dataset = exp_dataset
    self.predict_fn = predict_fn

    N, row, col, ch = self.exp_data.shape
    self.data_shape = self.exp_data[0].shape
    self.l1_avg = np.sum(self.exp_data / 255) / N
    print("l1 normalized avg {:.2f}".format(self.l1_avg))

    self.l1_bins = [0.02, 0.05, 0.1, 0.2]
    self.epsilons = [0.05, 0.1, 0.15, 0.3, 0.5, 0.75, 1, 2]
    self.fractions = [0.1, 0.2, 0.3, 0.4, 0.6]

    self.misclassified_data = {
      'anchor': {
        'p': [],
        'data': [],
        'original_data_idx': [],
        'original_labels': [],
        'labels': []
      },
      'anchor_comp': {
        'p': [],
        'data': [],
        'original_data_idx': [],
        'original_labels': [],
        'labels': []
      },
      'F': 0
    }
  
  def AddNoiseForMisclassification(self, noise_type="gauss"):
    num_misclassified_within_bounds = 0
    N, row, col, ch = self.exp_data.shape
    for i in range(N):
      anchor_data = {
        # w.r.t to true data point
        'l1_val': 9999999999,
        'label': None,
        'misclassified_image': None
      } 
      anchor_comp_data = {
        # w.r.t to true data point
        'l1_val': 9999999999,
        'label': None,
        'misclassified_image': None
      } 
      ##
      def sortSecond(val): 
        return val[1]
      
      if self.predict_fn is None and self.exp_dataset != "imagenet":
        raise Exception("Error: need prediction function of any model."\
          "\nHint: Use predict_fn.py or put dataset as imagenet for Inception")

      if self.exp_dataset == "imagenet":
        K.clear_session()
        model = InceptionV3(weights='imagenet')
        self.predict_fn = model.predict
      
      # True labels
      true_labels = self.exp_metadata['labels'][i]
      true_labels.sort(key = sortSecond, reverse=True)
      ##
      print("Adding noise to data point {}".format(i))
      start_time = timeit.default_timer()
      misclassfied = False
      for frac in self.fractions:
        for eps in self.epsilons:
          print("  Adding noise at fraction: {:.2f}, eps: {:.2f}....".format(
            frac, eps))
          noisy_anchor, noisy_anchor_comp = self.AddNoise(i, frac, noise_type)
          print("  Finished adding noise.")
          # Make sure it is normalized
          adv_anchor = self.exp_data[i] / 255 + eps * noisy_anchor
          adv_anchor_comp = self.exp_data[i] / 255 + eps * noisy_anchor_comp

          ### TODO: Make it work fast
          l1_anchor = np.sum(abs(adv_anchor - self.exp_data[i] / 255))
          l1_anchor_comp = np.sum(abs(adv_anchor_comp - self.exp_data[i] / 255))

          bound = self.l1_bins[len(self.l1_bins) - 1] * self.l1_avg

          print("l1 anchor: {:.2f} l1 anchor comp: {:.2f}".format(l1_anchor, l1_anchor_comp))
          if (( l1_anchor > bound and l1_anchor_comp > bound) or 
              (l1_anchor > anchor_data['l1_val'] and l1_anchor_comp > anchor_comp_data['l1_val'])):
            print("  early exit !!")
            continue
          ###
          
          if self.exp_dataset == "imagenet" or len(true_labels) == 1000:
            # default top classes is 3, but we need all 
            # so had to make it as different case
            tmp_adv_anchor_label = decode_predictions(
              self.predict_fn(adv_anchor.reshape(1, row, col, ch)),
              top=1000)
            tmp_adv_anchor_comp_label = decode_predictions(
              self.predict_fn(adv_anchor_comp.reshape(1, row, col, ch)),
              top=1000)
            
            adv_anchor_label, adv_anchor_comp_label = [], []
            for j in range(1000):
              adv_anchor_label.append(tmp_adv_anchor_label[0][j][1:])
              adv_anchor_comp_label.append(tmp_adv_anchor_comp_label[0][j][1:])
          
          else:
            tmp_adv_anchor_label =  self.predict_fn(
            adv_anchor.reshape(1, row, col, ch))
            tmp_adv_anchor_comp_label =  self.predict_fn(
              adv_anchor_comp.reshape(1, row, col, ch))
            
            adv_anchor_label, adv_anchor_comp_label = [], []
            for j in range(len(true_labels)):
              adv_anchor_label.append([str(j), tmp_adv_anchor_label[0][j]])
              adv_anchor_comp_label.append([str(j), tmp_adv_anchor_comp_label[0][j]])


            # Sorting labels as per highest confidence
            adv_anchor_label.sort(key = sortSecond, reverse=True)
            adv_anchor_comp_label.sort(key = sortSecond, reverse=True)

          # Check if mis classified
          if adv_anchor_label[0][0] != true_labels[0][0]:
            misclassfied = True
            print("  misclassification")
            if l1_anchor < anchor_data['l1_val']:
              print("    mc a")
              anchor_data['l1_val'] = l1_anchor
              anchor_data['label'] = adv_anchor_label
              anchor_data['misclassified_image'] = adv_anchor

          if adv_anchor_comp_label[0][0] != true_labels[0][0]:
            misclassfied = True
            if l1_anchor_comp < anchor_comp_data['l1_val']:
              print("    mc ac")
              anchor_comp_data['l1_val'] = l1_anchor_comp
              anchor_comp_data['label'] = adv_anchor_comp_label
              anchor_comp_data['misclassified_image'] = adv_anchor_comp

      if misclassfied == True:
        # Doesn't mean we have to include this example
        # Need to check if l1_val falls within bounds
        misclassfied_within_bounds_anchor = False
        misclassfied_within_bounds_anchor_comp = False
        print("true mc")
        if anchor_data['l1_val'] < 9999999999:
          prec = anchor_data['l1_val'] / self.l1_avg
          for bin in self.l1_bins:
            if prec <= bin:
              misclassfied_within_bounds_anchor = True
              print("  true mc a")
              self.misclassified_data['anchor']['p'].append(bin)
              self.misclassified_data['anchor']['data'].append(
                anchor_data['misclassified_image'])
              self.misclassified_data['anchor']['labels'].append(
                anchor_data['label'])
              self.misclassified_data['anchor']['original_labels'].append(
                true_labels)
              self.misclassified_data['anchor']['original_data_idx'].append(i)
              break
        
        if anchor_comp_data['l1_val'] < 9999999999:
          prec = anchor_comp_data['l1_val'] / self.l1_avg
          for bin in self.l1_bins:
            if prec <= bin:
              misclassfied_within_bounds_anchor_comp = True
              print("  true mc ac")
              self.misclassified_data['anchor_comp']['p'].append(bin)
              self.misclassified_data['anchor_comp']['data'].append(
                anchor_comp_data['misclassified_image'])
              self.misclassified_data['anchor_comp']['labels'].append(
                anchor_comp_data['label'])
              self.misclassified_data['anchor_comp']['original_labels'].append(
                true_labels)
              self.misclassified_data['anchor_comp']['original_data_idx'].append(i)
              break
        
        if misclassfied_within_bounds_anchor or misclassfied_within_bounds_anchor_comp:
          num_misclassified_within_bounds += 1
          print("found misclassification")
          if not(misclassfied_within_bounds_anchor):
            self.misclassified_data['anchor']['p'].append(-1)
            self.misclassified_data['anchor']['data'].append(
              anchor_data['misclassified_image'])
            self.misclassified_data['anchor']['labels'].append([])
            self.misclassified_data['anchor']['original_labels'].append(
                true_labels)
            self.misclassified_data['anchor']['original_data_idx'].append(i)
          
          if not(misclassfied_within_bounds_anchor_comp):
            self.misclassified_data['anchor_comp']['p'].append(-1)
            self.misclassified_data['anchor_comp']['data'].append(
              anchor_comp_data['misclassified_image'])
            self.misclassified_data['anchor_comp']['labels'].append([])
            self.misclassified_data['anchor_comp']['original_labels'].append(
                true_labels)
            self.misclassified_data['anchor_comp']['original_data_idx'].append(i)
        
      print("finished. Took {:.2f} sec\n\n\n".format(
        timeit.default_timer() - start_time))
    self.misclassified_data['F'] = num_misclassified_within_bounds

    return self.misclassified_data

  def AddNoise(self, n, frac, noise_type):
    
    if noise_type == "gauss":
      noisy_anchor = np.zeros(self.data_shape)
      noisy_anchor_comp = np.zeros(self.data_shape)
      row, col, ch = noisy_anchor.shape

      ## For fast computation
      anchor_pts = {}
      for pt in self.exp_metadata['explanation_anchor'][n]:
        anchor_pts[(pt[0], pt[1])] = 1

      mean = 0
      var = 1
      sigma = var**0.5
      # print("    Adding noise to anchor...")
      # Generating noise to anchor
      for pt in self.exp_metadata['explanation_anchor'][n]:
        if np.random.choice([True, False], size=(1), p=[frac, 1 - frac]):
          noisy_anchor[pt[0]][pt[1]] = np.random.normal(mean,sigma,(ch))
      # print("    Finished")

      # Generating noise to anchor's complemet
      # print("    Adding noise to anchor's complement...")
      for r in range(row):
        for c in range(col):
          if not((r, c) in anchor_pts):
            if np.random.choice([True, False], size=(1), p=[frac, 1 - frac]):
              noisy_anchor_comp[r][c] = np.random.normal(mean,sigma,(ch))
      # print("    Finished")
      return noisy_anchor, noisy_anchor_comp

  def GiveVisual(self, num=2, filename=None):
    fig, ax = plt.subplots(num, 4, figsize=(32, 6 * num))
    ax = ax.ravel()

    for i in range(num):
      # Show original image
      plt.sca(ax[i * 4])
      orig_data = self.exp_data[
        self.misclassified_data['anchor']['original_data_idx'][i]]

      true_labels = self.exp_metadata['labels'][
        self.misclassified_data['anchor']['original_data_idx'][i]]
      # Sorting labels as per highest confidence
      def sortSecond(val): 
        return val[1]
      true_labels.sort(key = sortSecond, reverse=True)
      
      plt.title('class: {} confi: {:.2f}'.format(
        true_labels[0][0], true_labels[0][1] * 100))
      
      plt.imshow(orig_data)

      # Show Anchor
      plt.sca(ax[i * 4 + 1])
      anchor = np.zeros((orig_data.shape))
      for pt in self.exp_metadata['explanation_anchor'][
        self.misclassified_data['anchor']['original_data_idx'][i]]:
        anchor[pt[0]][pt[1]] = orig_data[pt[0]][pt[1]]
      
      plt.title('anchor')
      plt.imshow(anchor)

      # Show anchor misclassification
      plt.sca(ax[i * 4 + 2])

      if self.misclassified_data['anchor']['p'][i] != -1:
        plt.title('anchor misclassifcation\nclass: {} confi: {:.2f}'.format(
          self.misclassified_data['anchor']['labels'][i][0][0],
          self.misclassified_data['anchor']['labels'][i][0][1] * 100
        ))

        plt.imshow(self.misclassified_data['anchor']['data'][i])      

      # Show anchor's complement misclassification
      plt.sca(ax[i * 4 + 3])
      if self.misclassified_data['anchor_comp']['p'][i] != -1:
        plt.title('anchors complement misclassifcation\nclass: {} confi: {:.2f}'.format(
          self.misclassified_data['anchor_comp']['labels'][i][0][0],
          self.misclassified_data['anchor_comp']['labels'][i][0][1] * 100
        ))

        plt.imshow(self.misclassified_data['anchor_comp']['data'][i])

    if filename is None:
      filename = "{}_{}_result.pdf".format(self.exp_metadata['method'], num)
    
    plt.savefig(filename)
    plt.close()

  