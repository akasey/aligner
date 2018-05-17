import matplotlib.pyplot as plt

ecoli_precision_1000 = [0.9813, 0.9462,0.8217,0.4169]
ecoli_recall_1000 = [0.9797, 0.9334, 0.5614, 0.0127]
ecoli_f1_1000 = [0.9805, 0.9397, 0.6671, 0.0247]
ecoli_sensitivity_1000 = [0.9982, 0.9862, 0.6832, 0.0305]

ecoli_precision_2000 = [0.9838, 0.9489, 0.8120, 0.3830]
ecoli_recall_2000 = [0.9778, 0.9375, 0.6316, 0.0154]
ecoli_f1_2000 = [0.9808, 0.9432, 0.7105, 0.0297]
ecoli_sensitivity_2000 = [0.9939, 0.9879, 0.7777, 0.0403]


aceto_precision_1000 = [0.9785, 0.9537, 0.8730, 0.5643]
aceto_recall_1000 = [0.8255, 0.4945, 0.0766, 0.0002]
aceto_f1_1000 = [0.8955, 0.6513, 0.1408, 0.0004]
aceto_sensitivity_1000 = [0.8437, 0.5185, 0.0877, 0.0003]

aceto_precision_2000 = [0.9670, 0.8858, 0.5933, 0.0549]
aceto_recall_2000 = [0.8779, 0.6898, 0.2326, 0.0052]
aceto_f1_2000 = [0.9203, 0.7756, 0.3342, 0.0095]
aceto_sensitivity_2000 = [0.9078, 0.7787, 0.3920, 0.0954]

x_ticks = ["0.02", "0.05", "0.10"]

plt.subplot(211)
plt.xlabel("Error Rate")
plt.ylabel("F1")
plt.title("Escherichia coli")
m_1000, = plt.plot(ecoli_f1_1000[0:3], 'bo-', label = "[1000]")
m_2000, = plt.plot(ecoli_f1_2000[0:3], 'ro-', label = "[2000,1000]")
plt.xticks(range(len(x_ticks)), x_ticks)
plt.legend([m_1000, m_2000], ["Model 1", "Model 2"])


plt.subplot(212)
plt.xlabel("Error Rate")
plt.ylabel("F1")
plt.title("Acetobacter pasteurianus")
m_1000, = plt.plot(aceto_f1_1000[0:3], 'bo-', label = "[1000]")
m_2000, = plt.plot(aceto_f1_2000[0:3], 'ro-', label = "[2000,1000]")
plt.xticks(range(len(x_ticks)), x_ticks)
# plt.legend([m_1000, m_2000], ["Model 1", "Model 2"])

plt.tight_layout()
plt.savefig("sample_classification_run/f1_plot.png")
plt.show()



plt.subplot(211)
plt.xlabel("Error Rate")
plt.ylabel("Sensitivity")
plt.title("Escherichia coli")
m_1000, = plt.plot(ecoli_sensitivity_1000[0:3], 'bo-', label = "[1000]")
m_2000, = plt.plot(ecoli_sensitivity_2000[0:3], 'ro-', label = "[2000,1000]")
plt.xticks(range(len(x_ticks)), x_ticks)
plt.legend([m_1000, m_2000], ["Model 1", "Model 2"])


plt.subplot(212)
plt.xlabel("Error Rate")
plt.ylabel("Sensitivity")
plt.title("Acetobacter pasteurianus")
m_1000, = plt.plot(aceto_sensitivity_1000[0:3], 'bo-', label = "[1000]")
m_2000, = plt.plot(aceto_sensitivity_2000[0:3], 'ro-', label = "[2000,1000]")
plt.xticks(range(len(x_ticks)), x_ticks)
# plt.legend([m_1000, m_2000], ["Model 1", "Model 2"])

plt.tight_layout()
plt.savefig("sample_classification_run/sensitivity_plot.png")