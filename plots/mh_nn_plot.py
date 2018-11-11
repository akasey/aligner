import numpy as np
import matplotlib.pyplot as plt
import math

# , 1, 1, 1, 1
# , 0.408, 0.285, 0.153, 0.081
precision = [0.01, 0.186, 1] + [0.0468903, 0.0539234, 0.0824988, 0.107244, 0.130958, 0.156394, 0.349833, 0.426043, 0.482984, 0.728655]
recall = [1, 0.591, 0.408] + [0.867347, 0.836735, 0.806122, 0.755102, 0.734694, 0.72449, 0.55102, 0.520408, 0.510204, 0.489796]

recall_sorted = sorted(recall)
precision_sorted = [x for _,x in sorted(zip(recall,precision))]

nn_precision = [0.959184]
nn_recall = [0.959184]

# precision = [0.001, 0.0019, 0.0411937, 0.678672, 0.857407, 0.906863, 0.927536]
# recall = [1, 0.6979, 0.46875, 0.46875, 0.427083, 0.333333, 0.229167]


fig, ax = plt.subplots()
mh_line = ax.plot(recall_sorted, np.array(precision_sorted), 'bx--', linewidth=0.6, label="Minhash")
nn_line = ax.plot(nn_recall, np.array(nn_precision), 'gs', label="Neural Network")
plt.xlabel("Recall")
plt.ylabel("Precision")

for r,p in zip(recall_sorted, precision_sorted):
    MCC = "{0:.2f}".format(math.sqrt(r*p))
    ax.annotate(MCC, (r+0.005,p+0.01))

ax.annotate("{0:.2f}".format(math.sqrt(nn_recall[0]*nn_precision[0])), (nn_recall[0]+0.005, nn_precision[0]+0.01))
plt.legend()

fig.savefig('/Users/akash/PycharmProjects/aligner/sample_classification_run' + '/precision_recall.png')
plt.show()