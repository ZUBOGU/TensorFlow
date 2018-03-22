# TensorFlow
## Note For TensorFlow
###Tensor
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

## Syntax Note
tensoflow operator or tensor (Placeholder)
```
tf.placeholder("float", name="house_size")
tf.placeholder("float", name="price")
```
tensor varibales
```
tf.Variable(np.random.randn(), name="size_factor")
tf.Variable(np.random.randn(), name="price_offset")
```
"+" and "*" operators.
```
tf.add()
tf.multiply()
tf.pow()
```
Gradient Descent Optimizer. Loss Function - Mean squared error
```
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples) // need this 2
tf_cost = tf.reduce_mean(tf.square(tf_price_pred-tf_price))
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)
```
Run the tensor
```
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```
