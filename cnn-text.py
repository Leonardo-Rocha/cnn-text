import tensorflow as tf
import numpy as np

class TextCNN(object):
  #Definimos sequence_lenght como o tamanho das frases da entrada input_x, num_classes como as possíveis labels;
  #vocab_size  como o número de palavras relevantes ao nosso problema +1; embedding_size como o tamanho dos word vectors;
  # filter_sizes como o número de word vectors que analisamos por filtro durante a convolution layer;
  def __init__(
    self, sequence_length, num_classes, vocab_size,embedding_size, filter_sizes, num_filters):
    
      # Placeholders para alocar espaço para o input, output e o dropout
      self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
      self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
