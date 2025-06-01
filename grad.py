# z = f(x,y) = x^1.1 - sin(x + 1.3*y)

import tensorflow as tf
import numpy as np
import matplotlib as plt
import pandas as pd

x = np.linspace(0.0, 4.0, 0.1)
x = tf.constant(x)

y = np.linspace(0.0, 4.0, 0.1)
y = tf.constant(y)

with tf.GradientTape() as tape:
  tape.watch(x)
  tape.watch(y)
  z = x ** 1.1 -np.sin(x + 1.3*y)

dz_dx = tape.gradient(z, x)

dz_dy = tape.gradient(z, y)

print(dz_dx.numpy())
print('\n')
print(dz_dy.numpy())
