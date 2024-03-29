#%% TensorFlow Basics: Getting Started With TensorFlow

#This gives Python access to all of TensorFlow's classes, methods, and symbols
import tensorflow as tf
tf.__version__
tf.executing_eagerly()

import numpy as np

# Some constants
node1 = tf.constant(1.0, dtype=tf.float32)
node2 = tf.constant(2.0) # also tf.float32 implicitly
print(node1, node2)

#Some computations by combining Tensor nodes
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("simple (+): ",(node1 + node2))

#%% Rank of tensors
scalar = tf.constant(100) # 0 dimension
print(scalar.get_shape())
print(scalar.shape)

vector = tf.constant([1,2,3,4,5]) # 1 dimension
print(vector.get_shape())

matrix = tf.constant([[1,2,3],[4,5,6]]) # 2 dimension
print(matrix.get_shape())
matrix.shape

cube_matrix = tf.constant([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]) # 3 dimension
print(cube_matrix.get_shape())

#%% Various Reduction's operation
#https://www.tensorflow.org/api_docs/python/tf

x = tf.constant([[1,3,5],[2,3,4]], dtype = tf.int32)
x
tf.reduce_max(x, axis = 0) # Row wise -> vertical downwards
tf.reduce_max(x, axis = 1) # Col wise -> horizontal
tf.reduce_max(x) # Overall in all direction

# CW: practice for tf.reduce_min, reduce_sum and tf.reduce_mean

#%% Few important operations
# Few more: tf.cast: Casts a tensor to a new type
tf.cast([1.2,2.8], tf.int32)
tf.cast([1.2,2.8], tf.float32)

# Few more: tf.reshape
tf.reshape(x, [-1]) # Reshape to a vector. -1 means all. Like np.ravel
tf.reshape(x, [3,2])

#%%TensorFlow inbuilt constants
tf.ones((3,2))
tf.zeros((2,2))

#tf.add(tf.ones(2,2), tf.constant(1)) #Follwoing will throw Error because of mismatch of dimension
tf.add(tf.ones((2,2)), tf.ones((2,2)))  # right
tf.ones(2,2) + tf.ones(2,2) # Same as above
x + x

tf.multiply(x,x)
x*x #s same as above
#%%Tensor segmentation
seg_ids = tf.constant([0,1,1,2,2]); # Group indexes : 0|1,2|3,4
x = tf.constant([[2, 5, 3, -5],[0, 3,-2, 5],[4, 3, 5, 3],[6, 1,4, 0],[6, 1, 4, 0]]) # A sample constant matrix
tf.math.segment_sum(x, seg_ids)
#array([[ 2, 5, 3, -5],
#[ 4, 6, 3, 8],
#[12, 2, 8, 0]])

tf.math.segment_prod(x, seg_ids)
#array([[ 2, 5, 3, -5],
#[ 0, 9, -10, 15],
#[ 36, 1, 16, 0]])

#CW: Run and see the output
tf.math.segment_min(x, seg_ids)
tf.math.segment_max(x, seg_ids)
tf.math.segment_mean(x, seg_ids)

#%% CW: Self practice of Sequences: Various sequence utilities
x = tf.constant([[2, 5, 3, -5],[0, 3,-2, 5],[4, 3, 5, 3],[6, 1, 4, 0]])

# argmin shows the index of minimum value of a dimension
tf.argmin(x, 1)

# argmax shows the index of maximum value of a dimension
tf.argmax(x, 1)

#where : Conditional operation on tensor
boolx = tf.constant([True,False]); a = tf.constant([2,7]); b = tf.constant([4,5])
tf.where(boolx, a+b, a-b)

#Make sure you have clear concept of sets theory studied in high school
#showing the intersection between lists
listx = tf.constant([1,2,3,4], shape = (1,4)) # ,5,6,7,8, 8
listy = tf.constant([4,5,8,9], shape = (1,4))

#The result is sparse tensor.
st_diff = tf.sets.difference(listx, listy, aminusb = True, validate_indices = False)
st_diff.values.numpy() # Extract info

#With monior twist of different dimension
listx = tf.constant([1,2,3,4,5,6,7,8, 8], shape = (1,9))
listy = tf.constant([4,5,8,9], shape = (1,4))

#The result is sparse tensor.
st_diff = tf.sets.difference(listx, listy, aminusb = True, validate_indices = False)
st_diff.values.numpy() # Extract info

# Similarly for intersection, union

#unique (showing unique values on a list).
listx = tf.constant([1,2,3,4,5,6,7,8, 8], shape = (9))
values, indexes = tf.unique(listx)
values.numpy()

#%% CW Practice:Tensor slicing and joining
#In order to extract and merge useful information from big datasets, the slicing and joining
#methods allow you to consolidate the required column information without having to
#occupy memory space with nonspecific information.

t_matrix = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
t_array = tf.constant([1,2,3,4,9,8,6,5])
t_array2= tf.constant([2,3,4,5,6,7,8,9])

#It extracts a slice of size size from a tensor input starting at the location specified by begin.
tf.slice(input_ =  t_matrix, begin = [1, 1], size = [2,2])

#Splits a tensor into sub tensors
split0, split1 = tf.split(value=t_array, num_or_size_splits=2, axis=0)

# View the splitted part's shape
tf.shape(split0)
tf.shape(split1)

# View the splitted part's content
split0
split1

# creates a new tensor by replicating input multiples times
tf.tile(input = [1,2], multiples = [3])

#Packs the list of tensors in values into a tensor with rank one higher than each tensor in values,
#by packing them along the axis dimension.
#Given a list of length N of tensors of shape (A, B, C); if axis == 0 then the output tensor will
#have the shape (N, A, B, C). if axis == 1 then the output tensor will have the shape (A, N, B, C).

# Simple example
tf.stack(values = [t_array, t_array2], axis=0)

# Detail example
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z], axis=0)  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
# Same as np.stack([x, y, z])

# unstack is just opposite of stack, explained above
tf.unstack(t_matrix)

# Concatenates the list of tensors values along dimension axis
tf.concat(values=[t_array, t_array2], axis=0,)

# One more detail example of concat
t1 = tf.constant([[1, 2, 3], [4, 5, 6]])
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])
tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
