import numpy as np
import sys
sys.path.insert(0,"../models/")
from SSD300_custom import SSD300

model = SSD300(return_base = False)

model.summary()

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

for idx, (name, weight) in enumerate(zip(names, weights)):
    print(idx,name, weight.shape)

print("***********")

for idx,layer in enumerate(model.layers):
	print(idx,layer.name)