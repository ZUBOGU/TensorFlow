import tensorflow as tf
import numpy as np
import math
import tflearn

#   Create data in straight Python, create some size(x) and price(y) data points, using price = (m * size) + b.  
#        Here b is a price base based on other factors

#  generation some house sizes between 1000 and 3500 (typical sq ft of house)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house )

# Generate house prices from house size with a random noise added+.
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)  

# you need to normalize values to prevent under/overflows.
def normalize(array):
    return (array - array.mean()) / array.std()

# 1. Get Data
# Split the data into training and testing, and normalized the data 

# define number of training samples, 0.7 = 70%.  We can take the first 70% since the values are randomized
num_train_samples = math.floor(num_house * 0.7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# One value, one value out
input = tflearn.input_data(shape=[None], name="input_data")
linear = tflearn.layers.core.single_unit(input, activation='linear', name ="Linear")

# define the optimizer
reg = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
						metric='R2', learning_rate=0.01, name="regression")

#   Fit/train the model
model = tflearn.DNN(reg)
model.fit(train_house_size_norm, train_price_norm, n_epoch=1000)

score = model.evaluate(test_house_size_norm, test_house_price_norm)
print("\nloss on test : {0}".format(score))



