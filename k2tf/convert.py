"""
This script converts a .h5 Keras model into a Tensorflow .pb file.

Attribution: This script was adapted from https://github.com/amir-abdi/keras_to_tensorflow

MIT License

Copyright (c) 2017 bitbionic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import os.path as osp
import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io


def convertGraph(modelPath, outputPath, prefix, name):
    """
    Converts an HD5F file to a .pb file for use with Tensorflow.

    Args:
        modelPath (str): path to the .h5 file
           outdir (str): path to the output directory
           prefix (str): the prefix of the output aliasing
             name (str):
    Returns:
        None
    """

    keras = tf.keras
    load_model = keras.models.load_model
    K = keras.backend

    os.makedirs(outputPath, exist_ok=True)

    K.set_learning_phase(0)

    net_model = load_model(modelPath)
    net_model.summary()

    numOutputs = net_model.output.shape[1]

    # Alias the outputs in the model - this sometimes makes them easier to access in TF
    pred = [None] * numOutputs
    pred_node_names = [None] * numOutputs
    for i in range(numOutputs):
        pred_node_names[i] = prefix+"_"+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    print("Output nodes names are: ", pred_node_names)

    sess = K.get_session()
    
    # Write the graph in human readable
    f = name + ".ascii"
    tf.train.write_graph(sess.graph.as_graph_def(), outputPath, f, as_text=True)
    print("Saved the graph definition in ascii format at: ", osp.join(outputPath, f))

    # Write the graph in binary .pb file
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, outputPath, name, as_text=False)
    print("Saved the constant graph (ready for inference) at: ", osp.join(outputPath, name))

