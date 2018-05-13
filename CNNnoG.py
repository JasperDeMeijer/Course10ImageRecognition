import numpy as np
import tensorflow as tf
import glob
from os import path, walk
from scipy.misc import imread
import time

shapex = 625
shapey = 532
channels = 1
stepsNN = 20


def cnn_fn(features, labels, mode):
    # input (numpy array filled with images with shape: shapex, shapey)
    input_layer = tf.reshape(features["x"], [-1, shapex, shapey, channels])

    # convolutional layer 1 (kernel size = 200x200)
    convo1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=2,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # pooling layer 1; 50 percent image size reduction
    pool1 = tf.layers.max_pooling2d(inputs=convo1, pool_size=[2, 2], strides=[2, 2])

    convo2 = tf.layers.conv2d(
        inputs=pool1,
        filters=2,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=convo2, pool_size=[2, 2], strides=[2, 2])

    flat = tf.reshape(pool2, [-1, int(shapex/4) * int(shapey/4) * 2])

    dense = tf.layers.dense(inputs=flat, units=1000, activation=tf.nn.relu)

    # no dropout
    drop = tf.layers.dropout(
        inputs=dense, rate=0.0,training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # seven classes
    logits = tf.layers.dense(inputs=drop, units=7)

    pred = {
        "classes" : tf.argmax(input=logits, axis=1),
        "probabilities" : tf.nn.softmax(logits, name="softmaxt")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=pred)

        # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=pred["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def runNN(train_data, eval_data, train_labels, eval_labels):
	cnn_classifier = tf.estimator.Estimator(
		model_fn=cnn_fn)

	#tensors_to_log = {"probabilities":"softmax_tensor"}
	#logging_hook = tf.train.LoggingTensorHook(
	#	tensors=tensors_to_log, every_n_iter=50)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":train_data},
		y=train_labels,
		batch_size=1,
		num_epochs=None,
		shuffle=True)
	start_time = time.time()
	cnn_classifier.train(
		input_fn=train_input_fn,
		steps=stepsNN)
	print("Time taken to train: "+str((time.time()-start_time)))
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x":eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)

	eval_results = cnn_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

def main():
	dataset = "processed/"
	kvasir_classes = next(walk(dataset))[1]
	kvasir_dict = {}
	print(kvasir_classes)
	pathlisttrain = []
	imglisttrain = []
	pathlisttest = []
	imglisttest = []


	for k_class in kvasir_classes:
		class_path = path.join(dataset, k_class)
		print(class_path)
		kvasir_dict[k_class] = []
		counter = 0
		class_name = kvasir_classes.index(k_class)
		print(class_name)
		for impath in glob.glob(path.join(class_path, '*.png')):
			if counter < 20:
				img = imread(impath, mode='L')
				img = np.divide(img, 255)
				print(img)
				imglisttest.append(img.astype(np.float64))
				pathlisttest.append(class_name)
				counter += 1
			else:
				img = imread(impath, mode='L')
				img = np.divide(img, 255)
				imglisttrain.append(img.astype(np.float64))
				pathlisttrain.append(class_name)
				counter += 1
	print(len(imglisttrain))

	imglisttrain = np.asarray(imglisttrain, dtype=np.float64)
	imglisttest = np.asarray(imglisttest, dtype=np.float64)

	pathlisttest = np.asarray(pathlisttest, dtype=np.int32)
	pathlisttrain = np.asarray(pathlisttrain, dtype=np.int32)
	runNN(imglisttrain, imglisttest, pathlisttrain, pathlisttest)

main()
