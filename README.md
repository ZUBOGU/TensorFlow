# TensorFlow
The learning notes for tensorflow and some sample projects.
## Notes For TensorFlow
An interface for expressing machine learning algorithms and an implementation for executing such algorithms.

A framework for creating ensemble algorithms for today’s most challenging problems.

Tensors flowing between Operations => “TensorFlow”

### Training a Model with TensorFlow
|Concept|Implementation|
|---|---|
|Prepared Data|Generated house size and price data|
|Inference|Price = (sizeFactor * size) + priceOffset|
|Loss Measurement|Mean Square Error|
|Optimizer to Minimize Loss|Gradient Descent Optimizer|

### Tensor
An n-dimensional array or list used in Tensor to represent all data.

Defined by the properties, Rank, Shape, and Type.

### Rank
Dimensionality of a Tensor
```
Rank Description Example
0 Scalar s = 145
1 Vector v = [1, 3, 2, 5, 7]
2 Matrix m = [ [1,5,6], [5,3,4] ]
3 3-Tensor (cube) c = [ [ [1,5,6], [5,3,4] ], [ [9,3,5], [3,4,9] ],[ [4,3,2], [3,6,7] ] ]
```
### Shapes
Shape of data in Tensor. Related to Rank.
```
Shape
[]
[5]
[2, 3]
[3, 2, 3]
```

### DataType
```
float32, float64
int8, int16, int32, int64 uint8, uint16
string
bool
complex64, complex128 qint8, qint16, quint8
```

### Methods
```
get_shape() – returns shape 
reshape() – changes shape 
rank – returns rank
dtype – return data type 
cast – change data type
```

## Debugging and Monitoring
```
Run in session context
- Python debugger limited usefulness
Session locality varies
- Local CPU
- Local GPU
- Remote System (GPU/CPU) 
- Remote Cluster
```
### Machine Learning Issues
```
Complex models 
Data and trends 
Long run times
```

### TensorFlow Features and Tools
#### Name 
```
tf_house_size = tf.placeholder("float", name="house_size")
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
```
#### Name Scope
```
with tf.name_scope("MNIST_Input"):
```

### Summary methods
```
tf.summary.scalar()
tf.summary.histogram()
tf.summary.merge_all()
tbWriter = tf.summary.FileWriter(logPath, sess.graph)
tbWriter.add_summary(summary,i)
```

### TensorBoard
Syntax to run tenserboard with log file
```
tensorboard --logdir=log_simple_graph # shell command
Go to specify url, i.e.
http://localhost:6006/
```

```
Visualizing Learning 
Visualize Computation Graph 
Monitor performance
Shows internal operations
Better Name Scope, better visualization
```
Adding Support for TensorBoard
```
Define log file location
Define names and name scopes 
Add Summary methods
Train
Run TensorBoard
```

### Neural Networks in TensorFlow
#### Linear Regression Example Revisited
Gradient Descent Optimizer. Loss Function - Mean squared error
```
tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size), tf_price_offset)
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples) // need this 2
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)
```

#### Neuron Architecture
```
(I1 * W1) + (I2 * W2) + … + (In * Wn) + Bias
Act((I * W) + Bias)
```
Inputs are from data. Weights are what we training on. Active function process the sum and produce the final Output in a neuron.

Neural Network Layers
```
Input Layer
Hidden Layer(s) 
Output Layer
```

### Training a Neural Network with TensorFlow
|Concept|Implementation|
|---|---|
|Prepared Data|MNIST Data|
|Inference|(x * weight + bias) -> activation|
|Loss Measurement|Cross Entropy|
|Optimizer to Minimize Loss|Gradient Descent Optimizer|

#### Handwritten digit recognition
MNIST dataset Classic testing set
```
yann.lecun.com/exdb/mnist
```

loss is cross entropy
```
cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

### Improving Performance by Going Deep
Deeper networks can learn more details. Simple to do with TensorFlow

|Concept|Implementation|
|---|---|
|Prepared Data|MNIST data and reshape as required|
|Inference|Matmul(Weight, x) + bias for entire NN|
|Loss Measurement|Cross Entropy|
|Optimizer to Minimize Loss|Gradient Descent Optimizer|

#### Deepening the Network
```
Consider data
- We have images!
Insert Convolution Network layers
- Inspects subsets of image
- Learns features of image
```

MNIST Inputs => Conv1 => Pool1 => Conv2 => Pool2 => Fully Connected => Output Neuron

Back Propagation => "BackProp",  and Forward Propagation

Could have overfitting: good at train data, bad at test data or real world.

By put bais on Fully Connected, reduce overfitting

## Inception v3
DeepMNIST  Only works with MNIST images.  Inception v3 is a model for image recognition.
```
Provided with TensorFlow
Superhuman image classification
- Human – 5.1% error
- Inception v3 - 3.46% error
Training takes a long time
- 2 weeks
- 8 NVIDIA K40 processors (GPUs designed for computation)
Trained on ImageNet dataset
```

### Transfer Learning
Check website
```
https://www.tensorflow.org/tutorials/image_retraining
```
Follow the instruction to re-training images.

Steps:
```
Get our data
Load Inception
Retrain next to last (Bottleneck) layer 
Replace last layer to output of our classes 
Evaluate the retrained model
```

## TensorFlowwith Add-ons
### Keras
```
Library on top of TensorFlow or Theano
Focused on Neural Networks
Soon part of TensorFlow
Expanding support to other frameworks
```
Using Keras
```
Define Model 
Add Layers
Compile 
Train 
Evaluate
```

## Syntax Notes
### tensor placeholder
```
tf.placeholder(
    dtype,
    shape=None,
    name=None
)
tf.placeholder(tf.float32, shape=[None, 784])
```
### tensor varibales
```
tf.Variable(<initial-value>, name=<optional-name>)
tf.Variable(tf.zeros([784, 10]))
```

### Operators
```
tf.add()
tf.multiply()
tf.pow()
tf.matmul()
tf.equal()
```

### Module: tf.nn 
Defined in ```tensorflow/python/ops/nn.py.``` Neural network support.

#### softmax
Computes softmax activations. ```softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)```
```
tf.nn.softmax(
    logits,
    axis=None,
    name=None,
    dim=None
)
logits: A non-empty Tensor. Must be one of the following types: half, float32, float64.
tf.nn.softmax(tf.matmul(x, W) + b)
```

Run the tensor
```
run(
    fetches,
    feed_dict=None,
    options=None,
    run_metadata=None
)
```
```
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
sess.close()
```
