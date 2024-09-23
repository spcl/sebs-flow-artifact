import os
import argparse

import numpy as np
import pandas as pd
import scipy

invos = []
invos.append([109.97, 215.42, 216.36, 217.32, 214.33]) #ault24
invos.append([92.87, 200.59, 202.07, 204.33, 201.86]) #ault01
invos.append([92.24, 199.26, 199.86, 202.99, 199.19]) #ault02
invos.append([94.37, 202.75, 202.69, 204.24, 201.63]) #ault03
invos.append([94.61, 201.62, 201.81, 203.69, 201.05]) #ault04



variations = []
for invo in invos:
    mini = min(invo)
    invo.remove(mini)
    variations.append(scipy.stats.variation(invo))
    print("variation: ", scipy.stats.variation(invo))
    #print(invos.get_group(key), "\n\n")

print("min variation: ", min(variations), "max: ", max(variations))
print("len", len(variations))
print("mean variation", sum(variations)/float(len(variations)))

#print("max: ", times.max(), "min: ", times.min())

#coeff = scipy.stats.variation(times)
#print(coeff)

#d_runtime = invos["end"].max() - invos["start"].min()
