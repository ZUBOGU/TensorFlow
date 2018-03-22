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
