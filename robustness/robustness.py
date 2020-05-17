import numpy as np

class Robustness:

  def __init__(self, misclassified_data, U=0.2, L=0.02):
    '''
    Class to give robustness score to explanation method/s
    misclassified_data
      Should be list containing the miscalssified data of 
      different method/s
    '''
    # Upper and Lower limit of l1 bounds
    self.U = U
    self.L = L
    self.misclassified_data = misclassified_data

    self.robustness_data = None

    if not(isinstance(self.misclassified_data, list)):
      raise Exception("misclassified_data should be of type list.")

    if len(self.misclassified_data) == 1:
      # return single roubstness score with metadata
      self.SingleMethodRobustness()
  
  def SingleMethodRobustness(self):
    robustness_val = 0

    robustness_scores = []
    bin_scores = []
    divergence_scores = []

    self.misclassified_data = self.misclassified_data[0]

    for i in range(self.misclassified_data['F']):
      if self.misclassified_data['anchor']['p'][i] == -1 and self.misclassified_data['anchor_comp']['p'][i] == -1:

        raise Exception("Something is wrong !!!!!!")

      if self.misclassified_data['anchor']['p'][i] == -1:
        robustness_scores.append(-1)
        bin_scores.append(-1)
        divergence_scores.append(-1)
      
      elif self.misclassified_data['anchor_comp']['p'][i] == -1:
        robustness_scores.append(1)
        bin_scores.append(1)
        divergence_scores.append(1)

      else:
        bin_score = (
          self.misclassified_data['anchor_comp']['p'][i] 
          - self.misclassified_data['anchor']['p'][i]) / (self.U - self.L)
        
        bin_scores.append(bin_score)

        divergence_score = self.Divergence(
          self.misclassified_data['anchor']['original_labels'][i],
          self.misclassified_data['anchor']['labels'][i],
          self.misclassified_data['anchor_comp']['labels'][i],
        )

        divergence_scores.append(divergence_score)

        robustness_scores.append(0.5 * bin_score + 0.5 * divergence_score)

    
    robustness_val = np.sum(np.asarray(robustness_scores)) / self.misclassified_data['F']

    self.robustness_data = {
      'F': self.misclassified_data['F'],
      'robustness_val': robustness_val,
      'robustness_scores': robustness_scores,
      'bin_scores': bin_scores,
      'divergence_scores': divergence_scores}
        
  
  def Divergence(self, true_labels, labels1, labels2):
    
    if len(labels1) != len(labels2):
      raise Exception("Something is wrong with labels !!!!!!")
    
    # sort original label w.r.t prediction. highest first
    def sortSecond(val): 
      return val[1]
    true_labels.sort(key=sortSecond, reverse=True)

    labels1_dict = {}
    labels2_dict = {}

    for i in range(len(labels1)):
      labels1_dict[labels1[i][0]] = labels1[i][1]
      labels2_dict[labels2[i][0]] = labels2[i][1]

    i = 1
    divergence1 = 0
    divergence2 = 0
    while True:
      if i >= len(true_labels) or true_labels[i-1][1] == 0:
        break
      divergence1 += i * abs(true_labels[i-1][1] - labels1_dict[true_labels[i-1][0]])
      divergence2 += i * abs(true_labels[i-1][1] - labels2_dict[true_labels[i-1][0]])
      i += 1

    divergence_diff = divergence2 - divergence1

    robustness = divergence_diff / (len(labels1) / 2)

    while robustness > 1:
      robustness /= 10
    # robustness_sign = robustness < 0
    # robustness = abs(robustness)  

     

    # # robustness should be b/w 0 and 1
    # while robustness != 0 and robustness * 10 <= 1:
    #   robustness *= 10

    # if robustness_sign:
    #   robustness *= -1
    
    return robustness


  def MakeReport(self, filename):
    report_txt_file = open(filename, 'w+') 

    report_txt_file.write('data points: {}\nrobustness: {:.2f}\n\n'.format(
      self.robustness_data['F'], self.robustness_data['robustness_val']))
    
    report_txt_file.write('===================================================================\n')

    report_txt_file.write('robustness_score   bin_scores    divergence_scores     p     q\n')

    report_txt_file.write('===================================================================\n')

    for i in range(self.robustness_data['F']):
      report_txt_file.write('{:.2f}               {:.2f}              {:.2f}         {}     {}\n'.format(
        self.robustness_data['robustness_scores'][i],
        self.robustness_data['bin_scores'][i],
        self.robustness_data['divergence_scores'][i],
        self.misclassified_data['anchor']['p'][i],
        self.misclassified_data['anchor_comp']['p'][i]
      ))

    report_txt_file.write('===================================================================\n')

    report_txt_file.write('{:.2f}               {:.2f}              {:.2f}\n'.format(
      np.mean(self.robustness_data['robustness_scores']),
      np.mean(self.robustness_data['bin_scores']),
      np.mean(self.robustness_data['divergence_scores'])
    ))

    report_txt_file.close()



