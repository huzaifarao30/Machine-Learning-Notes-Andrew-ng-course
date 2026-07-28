"""
NumPy and Python Review Lab. Week 2, Andrew Ng ML Specialization.
Full explanations kept, organized by section, nothing shortened.
"""

import numpy as np
import time


# =========================================================
# SECTION 3.3 VECTOR CREATION
# =========================================================

a = np.zeros(4)
print(a)
print(a.shape)
print(a.dtype)
# Why (4,) and not something like (4,1) or just 4?
# shape is always a tuple. For a 1D array with 4 elements, the shape is (4,),
# note the comma. That comma matters: it's Python's way of saying "this is a
# tuple with one item," not just the number 4. If you ever see (4,1) instead,
# that would mean a 2D array (a matrix) with 4 rows and 1 column, a totally
# different shape, even though it "contains the same numbers."
#
# Why float64 and not int? Even though the values are 0, NumPy defaults to
# floating point unless you tell it otherwise, because in ML almost everything
# you compute (weights, gradients, averages) needs decimals, not just whole numbers.

a = np.array([5, 4, 3, 2])
print(a.dtype)
# All 4 elements are whole numbers, so NumPy infers int64.

b = np.array([5., 4, 3, 2])
print(b.dtype)
# Only the first element (5.) has a decimal point, but NumPy arrays must hold
# ONE single dtype for all elements (the rule is: elements of a vector are all
# the same type). Since one element is a float, NumPy "upcasts" the WHOLE array
# to float64 so nothing gets lost. An int can become a float safely, but not the
# other way around, since a float like 5.7 can't be squeezed into an int without
# losing the .7. So b.dtype ends up float64.

a = np.arange(4.)
print(a)
print(a.shape)
print(a.dtype)
# np.arange(4.) works like Python's range(4), giving [0, 1, 2, 3], but since we
# wrote "4." with a decimal point, NumPy treats the stop value as a float, so ALL
# elements come out as floats too (same "one dtype for the whole array" rule).
# Output:
# [0. 1. 2. 3.]
# (4,)
# float64
# Compare: if you write np.arange(4), no decimal, you'd get [0 1 2 3], shape (4,),
# dtype int64. Same numbers, but int64 dtype, because nothing signaled "float."

a = np.random.rand(4)
print(a)
print(a.shape)
print(a.dtype)
# np.random.rand(4) generates 4 random numbers between 0 and 1. Random numbers
# are inherently decimals, so this always outputs float64, regardless of what
# you pass in, similar to np.zeros, this function always defaults to float.
# Output will look something like this (yours will differ, it's random):
# [0.5488135  0.71518937 0.60276338 0.54488318]
# (4,)
# float64


# =========================================================
# SECTION 3.4.1 INDEXING (1D)
# =========================================================

a = np.arange(10)
print(a)
# a = [0 1 2 3 4 5 6 7 8 9]
# arange(10) gives 10 numbers, but starting from 0, and it stops before
# reaching 10, just like Python's range. So it never includes the stop value.

print(a[2])
# 2, because indexing starts at 0, so a[2] is the 3rd element (index 0, 1, 2).
# It's a coincidence here that index equals value, since arange starts at 0.

print(a[-1])
# 9, negative indexing counts from the end. -1 means the last element, -2 would
# mean second to last, and so on. This saves you from writing a[len(a)-1] every
# time you want the last element.

try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
# There is no index 10 in this array since it only goes 0 to 9. What you get is
# an IndexError, something like:
# IndexError: index 10 is out of bounds for axis 0 with size 10
# Why not a segmentation fault: Python and NumPy do bounds checking before
# letting you access memory. They check "does this array actually have an
# index 10?" before touching anything. If not, they raise a clean Python
# exception you can catch, instead of crashing the whole program or corrupting
# memory, which is what a segfault would mean, accessing memory you shouldn't
# at the raw hardware level, like in C. This is why the try/except block
# exists: you wrap risky operations in try/except so your program can catch the
# error and handle it gracefully instead of crashing entirely.


# =========================================================
# SECTION 3.4.2 SLICING (1D)
# a[start:stop:step], stop is ALWAYS exclusive (never included)
# =========================================================

a = np.arange(10)
print(f"a = {a}")
# a = [0 1 2 3 4 5 6 7 8 9]

c = a[2:7:1]
print(c)
# c = [2 3 4 5 6]
# start at index 2, stop before index 7, step by 1 each time, so it grabs
# indices 2,3,4,5,6, NOT 7, since stop is exclusive, same rule as arange/range.

c = a[2:7:2]
print(c)
# c = [2 4 6]
# start at index 2, stop before index 7, step by 2 each time. Indices visited:
# 2, then 2+2=4, then 4+2=6, then 6+2=8, but 8 is NOT less than 7 (stop), so it
# stops there. So we land on indices 2, 4, 6, values [2, 4, 6].

c = a[3:]
print(c)
# c = [3 4 5 6 7 8 9]
# leaving out "stop" means "go all the way to the end of the array." Leaving
# out "step" defaults to 1 (move one at a time). So this grabs everything from
# index 3 onward.

c = a[:3]
print(c)
# c = [0 1 2]
# leaving out "start" means "start from the very beginning (index 0)." stop=3
# means "stop before index 3" (exclusive, same rule as always). Step defaults
# to 1 since none is given. So this grabs indices 0, 1, 2, values [0, 1, 2].

c = a[:]
print(c)
# c = [0 1 2 3 4 5 6 7 8 9]
# leaving out start, stop, and step all at once means "give me everything,
# unchanged." Start defaults to 0, stop defaults to the very end, step defaults
# to 1. So a[:] is basically a full copy of the whole array.


# =========================================================
# SECTION 3.4.3 SINGLE VECTOR OPERATIONS
# =========================================================

a = np.array([1, 2, 3, 4])
print(f"a             : {a}")
# a = [1 2 3 4]

b = -a
print(f"b = -a        : {b}")
# b = [-1 -2 -3 -4]
# negation flips the sign of EVERY element, element wise, not a single number.

b = np.sum(a)
print(f"b = np.sum(a) : {b}")
# b = 10
# adds up all elements (1+2+3+4=10), returns a single scalar, not an array.

b = np.mean(a)
print(f"b = np.mean(a): {b}")
# b = 2.5
# average of all elements (sum/count = 10/4 = 2.5), also a scalar.

b = a**2
print(f"b = a**2      : {b}")
# b = [1 4 9 16]
# this is element wise power, EACH element gets squared individually
# (1^2=1, 2^2=4, 3^2=9, 4^2=16). This is NOT matrix multiplication, just
# per element math.
#
# Key takeaway for all of these: operations like -a and a**2 act element wise
# (touch every element individually), while np.sum(a) and np.mean(a) collapse
# the whole array into one number. That distinction (element wise vs
# collapsing to scalar) will matter a lot later when you're computing cost
# functions.


# =========================================================
# SECTION 3.4.4 VECTOR VECTOR ELEMENT WISE OPERATIONS
# =========================================================

a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"Binary operators work element wise: {a + b}")
# a + b = [0 0 6 8]
# element wise means index to index: a[0]+b[0], a[1]+b[1], etc.
# (1 + -1 = 0), (2 + -2 = 0), (3 + 3 = 6), (4 + 4 = 8)
# NOT summed together into one number, this returns a full vector, same shape
# as the inputs.

c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)
# This will FAIL and raise an error, something like:
# ValueError: operands could not be broadcast together with shapes (4,) (2,)
# element wise operations need BOTH vectors to be the exact same length. a has
# 4 elements, c has 2, so there's no valid pairing for index 2 and 3, so NumPy
# refuses.
#
# Note: this shape matching rule is about how many elements each array has,
# not what dtype they are. NumPy is totally happy adding an int array to a
# float array element by element, it just automatically upcasts the result to
# float64 (same "safer, more general type" rule from earlier). So to be
# precise: different shapes cause an error, because there's no way to pair up
# every element. Different dtypes with the same shape are totally fine, NumPy
# just converts to the more general type for the result.


# =========================================================
# SECTION 3.4.5 SCALAR VECTOR OPERATIONS
# =========================================================

a = np.array([1, 2, 3, 4])

b = 5 * a
print(f"b = 5 * a : {b}")
# b = [5 10 15 20]
# a single number (scalar) gets multiplied against EVERY element of the
# vector. This is different from vector vector ops, here there's only ONE
# number involved, not a second array, so there's no shape matching needed,
# the scalar just applies to all.


# =========================================================
# SECTION 3.4.6 VECTOR VECTOR DOT PRODUCT
# The most important operation in this lab, it's the core building block of
# every prediction your model will ever make (f(x) = w dot x + b).
# What the dot product actually does: multiply corresponding elements, then
# add up all those products into a single number.
# =========================================================

def my_dot(a, b):
    """
    Compute the dot product of two vectors

    Args:
      a (ndarray (n,)):  input vector
      b (ndarray (n,)):  input vector with same dimension as a

    Returns:
      x (scalar):
    """
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")
# my_dot(a, b) = 24
# (1*-1) + (2*4) + (3*3) + (4*2) = -1 + 8 + 9 + 8 = 24
# Dot product is a combination of two things learned earlier: element wise
# multiply (like a * b, which gives a vector [-1, 8, 9, 8]), then np.sum() on
# that result (collapses it to one scalar). So dot product is basically
# np.sum(a * b) in spirit, except np.dot() does both steps in one optimized
# operation instead of two separate ones.

c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape}")
# c = 24
# c.shape = ()
# () is an EMPTY tuple, it means "zero dimensions." This is NumPy's way of
# representing a pure SCALAR (a single number), as opposed to a 1D array with
# one element, which would show shape (1,). So np.dot on two 1D vectors always
# collapses down to a 0D scalar result, NOT an array, same as our my_dot()
# function, which also just returns a plain number.
#
# Shape () is zero dimensions, meaning "just a bare number," no array wrapper
# at all, like 24 sitting there on its own, no brackets. This is what
# np.dot() on two 1D vectors gives you.
# Shape (1,) is one dimension, but that dimension has exactly 1 element in
# it. So it's still an array, just a tiny one, like [24] with brackets. It has
# an index (a[0] would work), whereas a true scalar with shape () has no
# index at all.
# Shape (4,) is one dimension, 4 elements, the normal vectors used above.
# (m, n) for 2D is rows times columns, m rows, n columns. Covered properly in
# the Matrices section below.

c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(b, a).shape = {c.shape}")
# c = 24
# c.shape = ()
# dot product is COMMUTATIVE for vectors, order doesn't matter, because you're
# multiplying the SAME pairs either way: a[0]*b[0] is the same as b[0]*a[0]
# (regular multiplication doesn't care about order), and addition doesn't
# care about order either.


# =========================================================
# SECTION 3.4.7 THE NEED FOR SPEED, vectorized vs for loop
# =========================================================

np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time
print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a, b)
toc = time.time()  # capture end time
print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a)
del(b)  # remove these big arrays from memory
# np.random.seed(1) locks the random number generator so you get the same
# "random" numbers every time you run this (makes results reproducible,
# useful for comparing runs).
# time.time() gives current time in seconds. You capture it before and after
# the operation, and the difference tells you how long it took.
# 1000*(toc-tic) converts seconds to milliseconds, just for readability.
# We run the SAME dot product two ways: once with np.dot (vectorized), once
# with your my_dot loop, same inputs, same math, same answer, but timed
# separately.
#
# The gap in duration is massive (in one real run, roughly 2000ms for the loop
# vs 5ms vectorized, about 400x faster). Why: the Python for loop in my_dot
# runs 10 million times, and every single iteration has Python interpreter
# overhead (checking types, managing memory, etc.) on top of the actual
# multiply, that overhead adds up brutally at this scale. np.dot() skips
# Python's loop entirely, it hands the whole array to optimized C code under
# the hood, which uses SIMD hardware to multiply and add many numbers per CPU
# cycle in parallel. This is exactly why every ML library (NumPy, PyTorch,
# TensorFlow) is built around vectorized operations instead of loops, at real
# dataset sizes a loop based implementation would be unusably slow.


# =========================================================
# SECTION 3.4.8 VECTOR VECTOR OPERATIONS IN COURSE 1
# Going forward, training data is stored in X_train, shape (m, n):
#   m = number of training examples (e.g. number of houses)
#   n = number of features (e.g. size, bedrooms, floors, age)
# X_train is a 2D array (matrix), NOT a 1D vector like used so far.
# w (weights) stays a 1D vector, shape (n,), one weight per feature.
# To predict for ONE example, you pull out ONE row: X_train[i]. Indexing into
# a 2D array like this drops it down to 1D, so X_train[i] comes out shape
# (n,), matching w's shape (n,). That's WHY np.dot(X_train[i], w) works, both
# sides are 1D vectors of the same length.
# =========================================================

X = np.array([[1], [2], [3], [4]])
# X is a matrix: 4 rows (4 examples), 1 column (1 feature each)
# X.shape = (4, 1)

w = np.array([2])
# w is a 1D vector: 1 weight, matching the 1 feature
# w.shape = (1,)

c = np.dot(X[1], w)
# X[1] grabs row at index 1, the SECOND row, since indexing starts at 0.
# X[1] = [2], this collapses from 2D down to 1D, shape (1,).
# np.dot([2], [2]) = 2*2 = 4, element wise multiply, then sum.

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")
print(c)
# X[1] has shape (1,)
# w has shape (1,)
# c has shape ()
# c = 4

# Real example, using the exact housing setup from the course notes: 4
# houses, each with 4 features (size, bedrooms, floors, age).
X_train = np.array([
    [2104, 5, 1, 45],   # house 0
    [1416, 3, 2, 40],   # house 1
    [1534, 3, 2, 30],   # house 2
    [852,  2, 1, 36]    # house 3
])
# X_train.shape = (4, 4), m=4 examples (houses), n=4 features each

w = np.array([0.1, 4, 10, -2])
# w.shape = (4,), one weight per feature:
# 0.1 for size, 4 for bedrooms, 10 for floors, -2 for age
b = 80  # base price

x_i = X_train[1]
# x_i = [1416, 3, 2, 40], pulled out ONE row, now it's 1D, shape (4,)

prediction = np.dot(x_i, w) + b
print(f"x_i = {x_i}")
print(f"prediction = {prediction}")
# x_i has shape (4,), w has shape (4,), same length, so the dot product pairs
# them up correctly:
# (1416*0.1) + (3*4) + (2*10) + (40*-2) + 80
# = 141.6 + 12 + 20 - 80 + 80
# = 173.6
#
# This is literally what happens every time your model makes a prediction:
# grab one house's row out of X_train (shape (n,)), dot it with w (also shape
# (n,)), add b, get a single predicted price.


# =========================================================
# SECTION 4.1 and 4.2 MATRICES, WHAT THEY ARE
# A matrix is just a 2D array, rows and columns. Same rule as before, all
# elements must be the same type. Notation wise, matrices get capital bold
# letters (like X), vectors get lowercase bold (like x).
# m = number of rows = number of training examples
# n = number of columns = number of features
# So a matrix of shape (m, n) is m houses, each with n features, literally
# the X_train used above.
# =========================================================


# =========================================================
# SECTION 4.3 MATRIX CREATION
# =========================================================

a = np.zeros((1, 5))
# np.zeros(shape) creates an array filled entirely with 0s. Shape here is a
# TUPLE (1, 5) instead of a single number like before (np.zeros(4)). Passing
# a tuple tells NumPy to make a 2D array: 1 row, 5 columns.
print(f"a shape = {a.shape}, a = {a}")
# a shape = (1, 5)
# a = [[0. 0. 0. 0. 0.]]

a = np.zeros((2, 1))
# same function, different shape tuple, 2 rows, 1 column each.
print(f"a shape = {a.shape}, a = {a}")
# a shape = (2, 1)
# a = [[0.]
#      [0.]]
# NumPy prints each ROW on its own line for 2D arrays, that's how you visually
# tell a 2D array apart from a 1D one when printed.

a = np.random.random_sample((1, 1))
# np.random.random_sample(shape) fills the array with random floats between 0
# and 1, same as the 1D version used earlier, just given a 2D shape tuple.
print(f"a shape = {a.shape}, a = {a}")
# a shape = (1, 1)
# a = [[0.548...]] (value will differ each run since it's random)
# shape (1,1) is a 2D array with ONE element, NOT the same as a plain scalar
# (shape ()) or a 1D vector with one element (shape (1,)).

a = np.array([[5], [4], [3]])
# np.array(nested_list) builds an array from values YOU provide directly.
# Double brackets means 2D: each inner [ ] is one ROW. Here: 3 rows, 1 column
# each (3 examples, 1 feature each).
print(f" a shape = {a.shape}, np.array: a = {a}")
# a shape = (3, 1)
# a = [[5]
#      [4]
#      [3]]

a = np.array([[5],   # you can also spread this across multiple lines
              [4],   # for readability, Python doesn't care about
              [3]])  # whitespace or newlines inside brackets, purely cosmetic
print(f" a shape = {a.shape}, np.array: a = {a}")
# a shape = (3, 1)
# a = [[5]
#      [4]
#      [3]]
# Same result as above, just written more readably across lines.


# =========================================================
# SECTION 4.4.1 INDEXING (Matrices)
# =========================================================

a = np.arange(6).reshape(-1, 2)
# np.arange(6) creates a 1D vector: [0 1 2 3 4 5]
# .reshape(-1, 2) reshapes that 1D vector into a 2D array with 2 COLUMNS. The
# "-1" tells NumPy: "figure out the number of ROWS yourself." Since there are
# 6 elements and we want 2 columns, NumPy computes rows = 6/2 = 3. So this is
# IDENTICAL to writing .reshape(3, 2), the -1 just saves you the math.
print(f"a.shape: {a.shape}, \na= {a}")
# a.shape = (3, 2)
# a = [[0 1]
#      [2 3]
#      [4 5]]

print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]}, type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")
# a[2, 0] means row index 2, column index 0. Row 2 is [4 5], column 0 of that
# row is 4. a[2,0] = 4. a[2,0].shape = (), a true scalar, zero dimensions,
# same rule as np.dot earlier. type is numpy.int64, a NumPy scalar type, not
# a plain Python int, but behaves similarly.

print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
# a[2] with only ONE index given means NumPy assumes you mean "row 2, ALL
# columns." a[2] = [4 5]. a[2].shape = (2,), this DROPS a dimension, a 2D row
# becomes a 1D vector (the exact same row pulling behavior from the
# X_train[i] example above).
#
# The rule to lock in: a[2, 0] (comma, two indices) grabs ONE number. a[2]
# (single index) grabs the ENTIRE row as a 1D vector. Same array, very
# different results depending on how many indices you give it.


# =========================================================
# SECTION 4.4.2 SLICING (Matrices)
# =========================================================

a = np.arange(20).reshape(-1, 10)
# np.arange(20) gives a 1D vector [0 1 2 ... 19]
# .reshape(-1, 10) reshapes into 2D with 10 columns, -1 means "figure out
# rows." 20 elements / 10 columns = 2 rows.
print(f"a = \n{a}")
# a =
# [[ 0  1  2  3  4  5  6  7  8  9]
#  [10 11 12 13 14 15 16 17 18 19]]
# shape = (2, 10)

print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")
# a[0, 2:7:1] means row 0 ONLY, then slice columns 2 to 6 (stop before 7),
# step 1. Row 0 = [0 1 2 3 4 5 6 7 8 9], columns 2:7 = [2 3 4 5 6]. Result
# shape = (5,), 1D, because we picked ONE specific row (index 0, not a
# range).

print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")
# a[:, 2:7:1] means the ":" for rows means "ALL rows," then slice columns 2:7
# on each. Row 0 columns 2:7 = [2 3 4 5 6]. Row 1 columns 2:7 = [12 13 14 15
# 16]. Result:
# [[ 2  3  4  5  6]
#  [12 13 14 15 16]]
# shape = (2, 5), stays 2D, because we kept BOTH rows (used ":" not a single
# index).
#
# To be precise about what's cut out: this is the SAME column slice (2:7)
# applied to both rows, giving you a smaller 2x5 rectangle cut out of the
# middle of the original 2x10 matrix. It is not "row 0 is one thing and row 1
# is another," both rows get the identical column range applied.

print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)
# a[:,:] means ":" for rows (all), ":" for columns (all), the entire matrix,
# unchanged. shape = (2, 10), same as original.

print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# a[1,:] means row index 1, ":" means all columns of that row.
# = [10 11 12 13 14 15 16 17 18 19]
# shape = (10,), 1D, because we picked ONE specific row.

print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
# a[1] gives the same result as a[1,:], when you only give ONE index on a 2D
# array, NumPy automatically assumes "give me all columns of that row." These
# two lines are just two different ways of writing the SAME thing.
#
# The rule to lock in from this whole section: whether a result stays 2D or
# collapses to 1D depends entirely on whether you picked ONE specific row
# (collapses to 1D) or used ":" to keep a range of rows (stays 2D). This is
# exactly the mechanism behind pulling X_train[i] out as a vector to dot with
# w.
#
# One more important note: slicing does NOT create a new independent array by
# default, it creates a "view" into the original. That means if you modify
# the sliced result, it can change the original array too. If you want an
# independent copy so changing one doesn't affect the other, write:
#   b = a[:, 2:7:1].copy()