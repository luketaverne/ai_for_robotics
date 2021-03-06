import FCLayer as fc_layer
import ActivationFunction as activation
import LossFunction as loss
import Support as sup
import numpy as np
import os
import pickle as pkl
import logging

logger = logging.getLogger('FCNetwork')
ch = logging.StreamHandler()
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(name)s.%(levelname)s: %(message)s')
ch.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(ch)

def parseLoglevel(loglevel_str):
  loglevel_str = loglevel_str.lower()
  if loglevel_str == 'error':
    return logging.ERROR
  elif loglevel_str == 'warn':
    return logging.WARN
  elif loglevel_str == 'info':
    return logging.INFO
  elif loglevel_str == 'debug':
    return logging.DEBUG
  else:
    raise Exception('Loglevel could not be set.')

class FCNetwork():
  '''
  Fully connected neural network.
  This is creating an output from an input x through FCLayers of arbitrary
  dimensions.
  The output activation is a sigmoid activation function per default.
  '''
  input_dim = 0
  output_dim = 0
  hidden_layer_specs = []
  layers = []

  def __init__(self, input_dim = 0, output_dim = 0, hidden_layer_specs = [], output_activation=activation.SigmoidActivation(), loglevel='info'):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_layer_specs = hidden_layer_specs
    self.layers = []
    logger.setLevel(parseLoglevel(loglevel))
    ch.setLevel(parseLoglevel(loglevel))

    if len(hidden_layer_specs) > 0:
      logger.info('Constructing network.')
      # First hidden layer, connected to input
      logger.info('  Constructing input layer with dimensions [{},{}]'.format(self.input_dim, hidden_layer_specs[0]['dim']))
      self.layers.append(fc_layer.FCLayer(0, hidden_layer_specs[0]['activation'],
                                          self.input_dim, hidden_layer_specs[0]['dim']))

      # Further hidden layers
      for i in range(1, len(self.hidden_layer_specs)):
        logger.info('  Constructing layer {} with dimensions [{},{}]'.format(i, hidden_layer_specs[i - 1]['dim'], hidden_layer_specs[i]['dim']))
        self.layers.append(fc_layer.FCLayer(i, hidden_layer_specs[i]['activation'],
                                            hidden_layer_specs[i - 1]['dim'],
                                            hidden_layer_specs[i]['dim']))

      # Output layer
      logger.info('  Constructing output layer with dimensions [{},{}]'.format(hidden_layer_specs[-1]['dim'], self.output_dim))
      self.layers.append(fc_layer.FCLayer(len(self.layers), output_activation, hidden_layer_specs[-1]['dim'], self.output_dim))

      logger.info('Done with the network construction.')

  def numberOfLayers(self):
    return len(self.layers)


  def printDimensions(self):
    for idx, l in enumerate(self.layers):
      if idx == len(self.layers) - 1:
        print('Output layer has dimension {}'.format(l.outputDimension()))
      else:
        print('Hidden layer {} has dimension {}'.format(idx + 1, l.outputDimension()))


  def evaluateLayer(self, idx_layer, x):
    '''
    Evlauate single (hidden) layer of the network, given the inputs.
    idx_layer: index of layer that should be evaluated (idx = 0 means input)
    x: input data vector
    '''
    # print("Number of layers: " + str(self.numberOfLayers()))
    # print(idx_layer)
    # print(self.layers[idx_layer].getWeights().shape)
    if idx_layer == 0:
      return x

    else:
      # TODO: evaluate layer with specific idx
      # self.layer[i] is type fc_layer
      prev_layer_output = x
      for i in range(1,idx_layer):
        #   print("Looping layer: " + str(i))
          prev_layer_output = self.layers[i - 1].output(prev_layer_output) # Recursive and probably inefficient


      return self.layers[idx_layer - 1].output(prev_layer_output)

  def output(self, x):
    return self.evaluateLayer(len(self.layers), x)


  def getParameters(self):
    nn_params = sup.Variables()
    for l in self.layers:
      nn_params.weights.append(l.getWeights())
      nn_params.biases.append(l.getBiases())
    return nn_params


  def setParameters(self, param_dict):
    assert len(param_dict.weights) == len(self.layers), "Length of provided weights and network weights don't fit ({} vs {})".format(len(param_dict.weights), len(self.layers))
    assert len(param_dict.biases) == len(self.layers), "Length of provided biases and network biases don't fit ({} vs {})".format(len(param_dict.biases), len(self.layers))

    for layer_idx in range(len(self.layers)):
      self.layers[layer_idx].W = param_dict.weights[layer_idx]
      self.layers[layer_idx].b = param_dict.biases[layer_idx]


  def gradients(self, x, loss_function, y_target):
    '''
    Inputs:
      x: network input
      loss_function: Since the gradients of the loss function need to be computed, this has to be provided.
      y_target: Target values for the network output.
    Return value:
      gradients: Gradients of the loss function w.r.t. all weights and biases of the network.
                 Gradients have a weights and biases member, the indexing starts with 0 for the first hidden layer (W_1, b_1)
                 and ends with the output layer (W_out, b_out)
    '''
    gradients = sup.Variables()
    gradients.weights = [None]*self.numberOfLayers() #Start with None, should definitely throw an error if we have an indexing problem later
    gradients.biases = [None]*self.numberOfLayers()

    # Outputs of each layer (layer_evaluations[0] is input x)
    layer_evaluations = []
    for layer_idx, layer in enumerate(self.layers):
      layer_evaluations.append(self.evaluateLayer(layer_idx, x))

    # Output equals the evaluation of the last layer
    network_output = self.output(x)
    print("Network output shape: {}".format(network_output))

    ## Output layer
    dCost_dy = loss_function.derivative(network_output, y_target) #TODO: implement cost function derivative w.r.t. output variables of network
    # Element-wise multiplication with sigmoid derivative (sigmoid is applied element-wise)
    # delta_fused = dCost_dy * self.layers[-1].derivativeActivation(network_output) #TODO: start backpropagating the error
    #TODO: the above line is more general, but does not work right now
    delta_fused = dCost_dy * network_output * (1 - network_output)
    # L_w_out = np.dot(network_output.T, delta_fused)
    # L_b_out = delta_fused
    print("Output weights shape: {}".format(np.dot(layer_evaluations[-1].T, delta_fused)))
    gradients.weights[self.numberOfLayers()-1] = np.dot(layer_evaluations[-1].T, delta_fused)
    gradients.biases[self.numberOfLayers()-1] = delta_fused

    print(layer_evaluations)

    # Gradient backpropagation
    ## Start from last layer and propagate error gradient through until first layer
    ## Attention!!!: layer_evaluations[0] is the network input while self.layers[0] is the first hidden layer
    for layer_idx in np.arange(len(self.layers)-2, -1, -1):
      logger.debug('Computing the gradient for layer {}'.format(layer_idx))
      print("Gradient for layer_idx: {}".format(layer_idx))
      # If layer is not last layer, update delta_fused (which is accumulating the back-propagated gradient)
      # TODO: implement backpropagation of gradient for arbitrary number of layers
      L_w_prev = self.layers[layer_idx+1].getWeights() # 'prev' w.r.t. the back prop
    #   print("Current layer evaluations shape: {}".format(layer_evaluations[layer_idx]))
      print("\tPrevious layer weights (index {}) shape: {}".format(layer_idx +1,L_w_prev.shape))
      print("\tPrevious layer biases (index {}) shape: {}".format(layer_idx+1,gradients.biases[layer_idx+1].shape))
      print("\tCurrent delta_fused shape: {}".format(delta_fused.shape))

    #   delta_fused = np.dot(delta_fused, L_w_prev.T) * self.layers[layer_idx].derivativeActivation(layer_evaluations[layer_idx])
      delta_fused = np.dot(delta_fused, L_w_prev.T) * layer_evaluations[layer_idx+1] * (1 - layer_evaluations[layer_idx +1])

      gradients.weights[layer_idx] = np.dot(layer_evaluations[layer_idx].T, delta_fused) # weight
      gradients.biases[layer_idx] = delta_fused # bias

      print("\tLayer index {} will have weight shape {} and bias shape {}".format(layer_idx,gradients.weights[layer_idx].shape,gradients.biases[layer_idx].shape))

    return gradients

  def saveModel(self, path, filename):
    full_path = os.path.join(path, filename)
    print('Saving model to "{}"'.format(full_path))
    if not os.path.isdir(path):
      os.mkdir(path)
    pkl.dump(self, open(full_path, 'wb'))

  def loadModel(self, path, filename):
    full_path = os.path.join(path, filename)
    print('Loading model from "{}"'.format(full_path))
    if os.path.exists(full_path):
      network_tmp = pkl.load(open(full_path, 'rb'))
      self.input_dim = network_tmp.input_dim
      self.output_dim = network_tmp.output_dim
      self.hidden_layer_specs = network_tmp.hidden_layer_specs
      self.layers = network_tmp.layers
    else:
      raise Exception('File "{}" does not exist.'.format(full_path))
