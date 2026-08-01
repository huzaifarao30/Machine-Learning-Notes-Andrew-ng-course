import numpy as np

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# data is stored in numpy array/matrix
print(f"X Shape: {x_train.shape}, X Type:{type(x_train)})")
print(x_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

# X_train shape: (3, 4), 3 houses (m=3), 4 features each (n=4): size, bedrooms, floors, age.
# y_train shape: (3,), the actual prices for those 3 houses.

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
# w_init shape (4,), one weight per feature, pre-chosen (near-optimal) for this demo.
# b_init, a single scalar, the bias.

x_vec = x_train[0]
print(x_vec)
print(x_vec.shape)
# x_train[0] grabs row 0 (house 0)'s full feature vector, collapses 2D down to 1D,
# shape (4,). Same rule from the numpy lab: one index on a 2D array pulls the whole row.


##################### Section 3.1 - Predict with a loop (the "long way") ###############################

def predict_single_loop(x, w, b):
    """
    single predict using linear regression

    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):  model parameter

    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    # n = number of features in this ONE example. x.shape[0] reads the shape
    # tuple of x (which is 1D here, shape (n,)), NOT actual feature data.
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        # multiply feature i by its matching weight i
        p = p + p_i
        # accumulate the running total across all features
    p = p + b
    # add the bias ONCE, after the loop over all features finishes
    return p

# So to restate cleanly: for each feature i, multiply that feature's value by
# its corresponding weight, add all those products together, then add b at
# the very end. That's the full linear regression formula, done manually with
# a loop.

x_vec = x_train[0, :]
# X_train[0, :] and X_train[0] give the IDENTICAL result. Giving only one
# index on a 2D array implicitly means "all columns", the ":" here is just
# being explicit about it.
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
# x_vec shape (4,), x_vec value: [2104 5 1 45]

f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
# f_wb shape (), prediction: 459.9999976194083
# shape is () because the final p is built from numpy scalar math (x[i] is a
# numpy.float64), so it still carries a .shape attribute, and for a single
# value that's always the empty tuple, zero dimensions, same rule as
# np.dot() on two 1D vectors from the earlier numpy lab.


##################### Section 3.2 - Predict with np.dot (the "short way") ###############################

def predict(x, w, b):
    """
    single predict using linear regression, vectorized version

    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):             model parameter

    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    # np.dot(x, w) does the same job as the whole loop in predict_single_loop:
    # multiplies each feature by its matching weight, then sums all of them.
    # Adding b at the end gives the final prediction, one line instead of a loop.
    return p


x_vec = x_train[0, :]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
# x_vec shape (4,), x_vec value: [2104    5    1   45]

f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
# f_wb shape (), prediction: 459.99999761940825
# Matches predict_single_loop's result almost exactly (same math, just vectorized)
# From here on, predict() replaces predict_single_loop() everywhere, real code
# uses the vectorized version, the loop version was only to see the mechanics.


##################### Section 4 - Compute Cost with Multiple Variables ###############################
# Formula: J(w,b) = (1/2m) * sum over all m examples of (prediction - actual)^2
#
# You loop m times (once per house). On each iteration, you do the full
# sequence: predict -> subtract target -> square that single error -> add it
# to a running total. So by the end of the loop, you have one running number
# (not an array), the sum of all m squared errors. THEN, after the loop
# finishes, you divide that one final number by 2m.

def compute_cost(x, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    # m = number of ROWS in x (number of training examples). x.shape returns
    # the tuple (m, n); [0] picks the first item of that tuple, this reads
    # dimension info, it does NOT grab any actual row's data.
    cost = 0.0
    for i in range(m):
        prediction = predict(x[i], w, b)
        # predict for house i, using the FULL weight vector w (not w[i],
        # every house uses the same complete set of weights)
        prediction = (y[i] - prediction)**2
        # y[i] is the single target for house i (not the whole y array),
        # subtract, then square this ONE error
        cost = cost + prediction
        # accumulate the raw squared errors first, don't divide yet
    cost = cost / (2 * m)
    # divide ONCE, after the loop, matches the formula's structure: sum
    # everything first, then apply the 1/2m factor outside the sum
    return cost

cost = compute_cost(x_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')
# Cost at optimal w : 1.5578904880036537e-12
# Extremely close to zero because w_init/b_init were specifically pre-chosen
# to already be near-optimal for this dataset, this call only MEASURES how
# good those given values are, it does not search for or find them.


##################### Section 5.1 - Compute Gradient with Multiple Variables ###############################
# Gradient descent needs to know, for each parameter, which direction to nudge
# it so the cost goes down. That direction comes from the derivative (slope)
# of the cost function:
#
# dJ/dw_j = (1/m) * sum over all m examples of (prediction - actual) * x_j
# dJ/db   = (1/m) * sum over all m examples of (prediction - actual)
#
# b is ONE number, so its gradient (dj_db) is also ONE number.
# w has n values (one per feature), so its gradient (dj_dw) needs n separate
# numbers, one per feature, since changing each weight affects the model
# differently.
#
# This needs a NESTED loop: an outer loop over houses (m of them), and inside
# that, an inner loop over features (n of them), because each house's error
# contributes to EVERY weight's gradient, not just one.
#
# dj_dw and dj_db work exactly like accumulators, same pattern as cost in
# compute_cost: each house ADDS its own contribution on top of whatever was
# already there from previous houses. Only at the very end, after ALL houses
# have added their piece, do you divide by m to get the average.

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = x.shape
    # unpacking: x.shape returns tuple (m, n), this grabs both numbers in one
    # line instead of writing m = x.shape[0] and n = x.shape[1] separately
    dj_dw = np.zeros((n,))
    # one "bucket" per feature, starts at 0, gets added to as we loop through houses
    dj_db = 0
    # single running total for b's gradient

    for i in range(m):
        # OUTER loop: one iteration per HOUSE
        error = (np.dot(x[i], w) + b) - y[i]
        # prediction for house i, minus house i's actual target
        # same error used in compute_cost, just NOT squared this time
        for j in range(n):
            # INNER loop: one iteration per FEATURE, runs for EVERY house
            dj_dw[j] = dj_dw[j] + error * x[i, j]
            # x[i, j] = house i's value for feature j (row i, column j)
            # add this house's contribution ONTO whatever was already
            # accumulated in dj_dw[j] from previous houses
        dj_db = dj_db + error
        # accumulate this house's error into the running total for b

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    # average across all m houses, same "accumulate first, divide once at the
    # end" pattern as compute_cost

    return dj_db, dj_dw

# Compute and display gradient
tmp_dj_db, tmp_dj_dw = compute_gradient(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
# dj_db at initial w,b: -1.6739251122999121e-06
# dj_dw at initial w,b:
#  [-2.73e-03 -6.27e-06 -2.22e-06 -6.92e-05]
# These gradients are tiny too, for the same reason cost was tiny: w_init and
# b_init are already near-optimal, so the "slope" telling us which way to
# adjust is barely there, we're already very close to the bottom of the bowl.


##################### Section 5.2 - Gradient Descent With Multiple Variables ###############################
# compute_gradient only gives ONE reading of the slope, at whatever w,b you
# hand it, it does NOT update w or b itself. gradient_descent is the function
# that actually repeats this process over and over, taking a step downhill
# each time, until w and b have been trained from scratch (starting at zero)
# into good values.
#
# The loop, in plain terms, repeated num_iters times:
#   1. get the current gradient (which way is downhill) at the current w, b
#   2. take a step in that direction, sized by alpha
#   3. optionally record the cost at this point, to see if it's shrinking

import copy, math

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)
    # w_in is passed by reference, w = w_in would just make w another name
    # pointing at the SAME array in memory, deepcopy makes a real,
    # independent copy so modifying w inside this function doesn't
    # accidentally modify the original w_in outside it
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)
        # gradient_function is whatever was PASSED IN when this function was
        # called (e.g. compute_gradient), called through this generic name
        # instead of the specific name so this function stays reusable

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # this is the actual "step downhill": nudge w and b in the direction
        # that reduces cost, alpha controls how big a step to take

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))
            # cost_function is the passed-in compute_cost, records how good
            # the model is right after THIS iteration's update, so we can
            # plot cost vs iteration afterward and see it shrink

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history  # return final w,b and J history for graphing


# initialize parameters
initial_w = np.zeros_like(w_init)
# np.zeros_like(w_init) creates an array of zeros with the SAME shape as
# w_init (4,), this is where we actually start training from scratch, unlike
# earlier sections where w_init/b_init were already given to us pre-solved
initial_b = 0.

# some gradient descent settings
iterations = 1000
alpha = 5.0e-7

# run gradient descent
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                             compute_cost, compute_gradient,
                                             alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m, _ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
# now w_final/b_final were ACTUALLY found by the algorithm, starting from
# zero, not handed to us pre-solved like w_init/b_init were earlier. These
# predictions won't be as accurate as the w_init ones since only 1000
# iterations were run, this is the real training process, not just a check.