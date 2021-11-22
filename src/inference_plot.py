import json
import matplotlib.pyplot as plt
import numpy as np

with open('inference_speed.json', 'r') as f:
    deltas = json.loads(f.read())

plt.boxplot(deltas)
plt.ylabel("Inference Speed (ms)")
plt.show()
