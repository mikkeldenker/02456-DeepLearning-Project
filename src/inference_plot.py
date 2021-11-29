import json
import matplotlib.pyplot as plt
import numpy as np

with open('inference_speed.json', 'r') as f:
    all_deltas = json.loads(f.read())

data = []
labels = []
for model in all_deltas:
    deltas = all_deltas[model]
    data.append(deltas)
    labels.append(model)
    print(model, np.median(all_deltas[model]))

plt.boxplot(data)
plt.ylabel("Inference Speed (sec)")
plt.xticks(list(range(1, len(data)+1)), labels)
plt.show()
