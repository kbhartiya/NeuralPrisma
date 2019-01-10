import tensorflow as tf
import numpy as np
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from nst_utils import *
import warnings; warnings.filterwarnings("ignore") 

def compute_content_cost(a_C, a_G):
	"""
	Computes the content cost
	
    Arguments:
	a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
	a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
	    
	Returns: 
	J_content.
	"""
	m, n_H, n_W, n_C = a_G.get_shape().as_list()
	a_C_unrolled = tf.transpose(tf.reshape(a_C,[-1]))
	a_G_unrolled = tf.transpose(tf.reshape(a_G,[-1]))
	norm = n_H*n_W*n_C
	J_content = 0.25*(1/norm)*tf.reduce_sum((a_C_unrolled-a_G_unrolled)**2)
	
	return J_content

def gram_matrix(A):
	"""
	Style Matrix
	------------
	Argument:
	A -- matrix of shape (n_C, n_H*n_W)
	
	Returns:
	GA -- Gram matrix of A, of shape (n_C, n_C)
	"""
	GM = tf.matmul(A,A,transpose_b=True)
	
	return GM
	
def compute_style_layer_cost(a_S, a_G):
	"""
	Computes the style cost for each layer.
	
	Arguments:
	a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of an image. 
	a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of generated image.
	
	returns:
	style_layer_cost.
	"""
	m, n_H, n_W, n_C = a_G.get_shape().as_list()
	
	a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C]))
	a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))
	
	GS = gram_matrix(a_S)
	GG = gram_matrix(a_G)

	style_layer_cost = tf.reduce_sum((GS-GG)**2)/(4*(n_C)**2*(n_H*n_W)**2)
	
	return style_layer_cost	
	
def compute_style_cost(model, style_layers, sess):
	"""
	Computes style cost for all the layers.
	
	Arguments:
	model -- model(in this case VGG19 model).
	style_layers -- List of tuples in which each tuple has:
					- The layers from which we want to extract our style from.
					- The corresponding coefficients for each layer.
	sess -- The session.
	
	returns:
	J_style.
	"""
	J_style = 0
	
	for layer, k in style_layers:
		out = model[layer]
		a_S = sess.run(out)
		a_G = out
		
		J_style += k*compute_style_layer_cost(a_S, a_G)
		
		return J_style
	

def total_cost(J_content, J_style, alpha=None, beta=None):
	return alpha*J_content + beta*J_style

#reset the graph
tf.reset_default_graph()

#Initiate an Interactive Session.
sess = tf.InteractiveSession()# error may occur so be careful.

#content_image
content_image = scipy.misc.imread("/images/contentsample.jpg")
content_image = reshape_and_normalize(content_image)

#style_image
style_image = scipy.misc.imread("/images/stylesample.jpg")
style_image = reshape_and_normalize(style_image)

#generate noisy image.
generate_image = generate_noise_image(content_image)

#Load the VGG19 model to model.
model = load_vgg_model("vggmodel/imagenet-vgg-verydeep-19.mat")
			
#pass the content image to the model.
sess.run(model['input'].assign(content_image))

#Now store the 'conv4_2' activation for the content image in 'out'.
out = model['conv4_2']

a_C = sess.run(out)
						
a_G = out

#Compute Content cost.
J_content = compute_content_cost(a_C, a_G)

#Style Layers.
STYLE_LAYERS = [('conv1_1', 0.1),('conv2_1', 0.1),('conv3_1', 0.1),('conv4_1', 0.3),('conv5_1', 0.4)]

#Pass the style image to the model.
sess.run(model['input'].assign(style_image))

#Compute style cost.
J_style = compute_style_cost(model, STYLE_LAYERS, sess)

#Compute the total cost.
J = total_cost(J_content, J_style, alpha=10, beta=50)

#Optimizer for training.
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations):
	
	#initialize all variables.
	sess.run(tf.global_variable_initializer())
	
	#input the randoml generated image to the VGG19 model.
	sess.run(model['input'].assign(input_image))
	
	for i in range(num_iterations):
		sess.run(train)
		#Compute the generated image by running it in the model.
		generated_image = sess.run(model['input'])
		
		if i%20:
			J, J_content, J_style = sess.run([J, J_content, J_style])
			print("Total Cost: {}".format(J))
			print("Total Content Cost: {}".format(J_content))
			print("Total Style Cost: {}".format(J_style))
			
			save_image('./output/generated_image_no'+str(i)+'.png', generated_image)
		
	save_image("./output/final_generated_image.jpg", generated_image)
	
	return generated_images	

		
	
	
	
	
			
