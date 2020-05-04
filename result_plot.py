import pickle
from datetime import date
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

experiment_prefix =path = "./experiments/05_03_citeseer_0.6_A-Meta-Train"
re_trainings = 20
variant = "A-Meta-Train"
share_perturbations=0.05
plot_perturbations = [0.05,0.1,0.2,0.3,0.4,0.5,0.6]

accuracies_clean_train= pickle.load(open( experiment_prefix + '/accuracies_clean_train.pickle', 'rb'))
accuracies_clean_test= pickle.load(open( experiment_prefix + '/accuracies_clean_test.pickle', 'rb'))
accuracies_clean_unlabeled= pickle.load(open( experiment_prefix + '/accuracies_clean_unlabeled.pickle', 'rb'))
accuracies_clean_val= pickle.load( open( experiment_prefix + '/accuracies_clean_val.pickle', 'rb'))


accuracies_pert_train = pickle.load( open( experiment_prefix + '/accuracies_pert_train.pickle', 'rb'))
accuracies_pert_test = pickle.load( open( experiment_prefix + '/accuracies_pert_test.pickle', 'rb'))
accuracies_pert_unlabeled = pickle.load(open( experiment_prefix + '/accuracies_pert_unlabeled.pickle', 'rb'))
accuracies_pert_val= pickle.load(open( experiment_prefix + '/accuracies_pert_val.pickle', 'rb'))

pert = np.repeat(plot_perturbations,len(accuracies_pert_test[0]))
p_test =  np.array(accuracies_pert_test).flatten()
p_val =  np.array(accuracies_pert_val).flatten()
p_train =  np.array(accuracies_pert_train).flatten()
p_unlabeled =  np.array(accuracies_pert_unlabeled).flatten()
c_test = np.tile(accuracies_clean_test,len(plot_perturbations))
c_train = np.tile(accuracies_clean_train,len(plot_perturbations))
c_val = np.tile(accuracies_clean_val,len(plot_perturbations))
c_unlabeled = np.tile(accuracies_clean_unlabeled,len(plot_perturbations))
data= pd.DataFrame({'pert':pert,"p_test":p_test, "c_test":c_test,
"p_val":p_val, "c_val":c_val,
"p_train":p_train, "c_train":c_train,
"p_unlabeled":p_unlabeled, "c_unlabeled":c_unlabeled
                    })

plt.figure()
sns.lineplot(x="pert", y="p_test", data=data, color="coral")
ax = sns.lineplot(x="pert", y="c_test", data=data, color="coral")
ax.lines[1].set_linestyle("--")
plt.title("Accuracy on test-set with "+ variant)
plt.xlabel("% edges perturbed")
plt.ylabel("accuracy")
plt.legend(labels=['perturbed graph', 'original graph'])
plt.savefig( experiment_prefix + "/plot_test.png", dpi=600)
plt.show()

plt.figure()
sns.lineplot(x="pert", y="p_val", data=data, color="blue")
ax = sns.lineplot(x="pert", y="c_val", data=data, color="blue")
ax.lines[1].set_linestyle("--")
plt.title("Accuracy on val-set with "+ variant)
plt.xlabel("% edges perturbed")
plt.ylabel("accuracy")
plt.legend(labels=['perturbed graph', 'original graph'])
plt.savefig( experiment_prefix + "/plot_val.png", dpi=600)
plt.show()



plt.figure()
sns.lineplot(x="pert", y="p_train", data=data, color="red")
ax = sns.lineplot(x="pert", y="c_train", data=data, color="red")
ax.lines[1].set_linestyle("--")
plt.title("Accuracy on train-set with "+ variant)
plt.xlabel("% edges perturbed")
plt.ylabel("accuracy")
plt.legend(labels=['perturbed graph', 'original graph'])
plt.savefig( experiment_prefix + "/plot_train.png", dpi=600)
plt.show()


plt.figure()
sns.lineplot(x="pert", y="p_unlabeled", data=data, color="purple")
ax  = sns.lineplot(x="pert", y="c_unlabeled", data=data, color="purple")
ax.lines[1].set_linestyle("--")
plt.title("Accuracy on test-set + val-set with "+ variant)
plt.xlabel("% edges perturbed")
plt.ylabel("accuracy")
plt.legend(labels=['perturbed graph', 'original graph'])
plt.savefig( experiment_prefix + "/plot_unlabeled.png", dpi=600)
plt.show()
