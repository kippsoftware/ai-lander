# Neural Network to Control Lunar Lander 

WORK IN PROGRESS

In 1990 I wrote software in Objective-C on the NeXT machine that
implemented a neural network to land a lunar lander module. The code
and paper were stored on a floptical drive, which failed many years
ago.

Today I bring the lander back to life and use it to explain machine
learning using neural networks.

## First Dimension: Vertical Descent

First consdier a one-dimensional lander flying above the lunar surface
(*y=0*). The lander has vertical position *y* and velocity *v = dy/dt*.

The lander has one thruster, say +1.80 m/s/s, that can accelerate the
lander opposite the pull of the moon's gravity, assumed to be -1.62
m/s/s.

In general terms, the desired behavior of the main thruster is this:

                 y
     +  -  -  -  |  -  -  -  -
                 |
     +  -  -  -  |  -  -  -  -
                 |
     +  +  -  -  |  -  -  -  -
                 |
     +  +  +  -  |  -  -  -  -
    ------------ v ------------

We want a slow, managed descent. Thus any time *v* is positive, we
turn off the thruster. When *v* is slightly negative, we float
downward. Only when downward velocity is too fast do we turn on the
thruster to avoid a crash. Ideally we land close to *y = 0* and *v =
-1m/s* and the simulation ends.

The one-dimensional lander has two inputs, altitude *y* and velocity *v*.
It has one output. The binary value for the thruster should be either on or off
depending on input, thus *t(y,v) in {-1,1}*.

We could write a quick algorithm to implement the above function.
However, when we add more degrees of freedom to the lander, the
algorithm gets complicated quickly!

Instead, let's use artificial intelligence to drive the lander.


## Training Data

Let's convert the graph above into numeric training data and testing data.

      y   v  | t
     ------------
      1  -3  | 1
      1  -2  | 1
      1  -1  | -1
      1   0  | -1
      1   1  | -1
      1   2  | -1
      etc.

Next we randomly partition the rows of data, putting the majority into
training data and the remainder into testing data (and, for production
systems, a third partition for validation data).

## Single Neuron

We make a simple neuron with two weights, feed in *y* and *v*, then compute

    t = tanh( y w0 + v w1 + b )

Here we weight the input position and velocity with *w0* and *w1*, add
some bias, then use hyperbolic tangent to squash the result into the
range (-1,1).

What weights should we choose? Will one neuron be enough?

Wait! How do we measure success? How do we know we won?


## Loss Function

Let's measure how far we are from success by squaring the difference
of what we wanted versus what we got. Squaring gives us a positive
statistic that we can later add to other errors in other dimensions.

    e = sum((t_desired - t_actual)**2)

We can be satisfied when e is below some tiny threshhold for all the
inputs in the test set.


## Visualization

Here is a web page including a visualizer, starting with the lander
above the moon, show it flying and possibly crashing. Initialize the
weights and bias to zero and watch the lander go. Fly away? Crash?

TBD: web simulator

OK, now set the weights and bias to random values. Fly away? Crash?

## Normalize the Inputs

    xs is the array of x values

Want inputs to distribute from (min(xs),max(xs)) to (-1,1).

    scale = max(x) - min(x)

    scaled_x = (1 - -1) * ((x - min(x)) / scale) + -1


## Ideal Weights

    t = tanh( y w0 + v w1 + b )

- Loop until error < threshold
  - Forward activation
  - Compute the loss
  - Backprop the gradients
  - Tweak each w0, w1, and b in the correct directions

Now fly in the simulator.
