#!/usr/bin/env python
# coding: utf-8

# In[1]:


from metattack import utils
from metattack import meta_gradient_attack as mtk
import numpy as np
import tensorflow as tf
import seaborn as sns
import pickle
from datetime import date
from matplotlib import pyplot as plt
import scipy.sparse as sp
import os
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

gpu_id = None


# Settings
# The attack variants from the paper
variants = ["Meta-Train", "Meta-Self","A-Meta-Train", "A-Meta-Self", "A-Meta-Both"]
# Choose the variant you would like to try
variant = "A-Meta-Train"
share_perturbations = 0.05
dataset = "citeseer"
train_iters = 100
experiment_prefix = "./experiments/"+date.today().strftime("%m_%d_")+dataset+"_"+str(share_perturbations)+"_"+variant
if not  os.path.isdir(experiment_prefix):
    os.makedirs(experiment_prefix)
record_experiment = True


# In[2]:

_A_obs, _X_obs, _z_obs = utils.load_npz('data/'+dataset+'.npz')
if _X_obs is None:
    _X_obs = sp.eye(_A_obs.shape[0]).tocsr()

_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1
lcc = utils.largest_connected_components(_A_obs)

_A_obs = _A_obs[lcc][:,lcc]
_A_obs.setdiag(0)
_A_obs = _A_obs.astype("float32")
_A_obs.eliminate_zeros()
_X_obs = _X_obs.astype("float32")

assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

_X_obs = _X_obs[lcc]
_z_obs = _z_obs[lcc]
_N = _A_obs.shape[0]
_K = _z_obs.max()+1
_Z_obs = np.eye(_K)[_z_obs]
_An = utils.preprocess_graph(_A_obs)
sizes = [16, _K]
degrees = _A_obs.sum(0).A1

seed = 15
unlabeled_share = 0.8
val_share = 0.1
train_share = 1 - unlabeled_share - val_share
np.random.seed(seed)

split_train, split_val, split_test= utils.train_val_test_split_tabular(np.arange(_N),
                                                                       train_size=train_share,
                                                                       val_size=val_share,
                                                                       test_size=unlabeled_share,
                                                                       stratify=_z_obs)

if record_experiment:
    pickle.dump(split_train, open(experiment_prefix+'/split_train.pickle', 'wb'))
    pickle.dump(split_val, open( experiment_prefix + '/split_val.pickle', 'wb'))
    pickle.dump(split_test, open( experiment_prefix + '/split_test.pickle', 'wb'))

split_unlabeled = np.union1d(split_val, split_test)


# In[3]:


hidden_sizes = [16]
perturbations = int(share_perturbations * (_A_obs.sum()//2))
dtype = tf.float32 # change this to tf.float16 if you run out of GPU memory. Might affect the performance and lead to numerical instability


# In[4]:


surrogate = mtk.GCNSparse(_A_obs, _X_obs, _Z_obs, hidden_sizes, gpu_id=gpu_id)
surrogate.build(with_relu=False)
surrogate.train(split_train)


# In[5]:


# Predict the labels of the unlabeled nodes to use them for self-training.
labels_self_training = np.eye(_K)[surrogate.logits.eval(session=surrogate.session).argmax(1)]
labels_self_training[split_train] = _Z_obs[split_train]


# In[6]:


assert variant in variants

enforce_ll_constrant = False
approximate_meta_gradient = False
if variant.startswith("A-"): # approximate meta gradient
    approximate_meta_gradient = True
    if "Train" in variant:
        lambda_ = 1
    elif "Self" in variant:
        lambda_ = 0
    else:
        lambda_ = 0.5
        
if "Train" in variant:
    idx_attack = split_train
elif "Self" in variant:
    idx_attack = split_unlabeled
else:  # Both
    idx_attack = np.union1d(split_train, split_unlabeled)

if record_experiment:
    pickle.dump(idx_attack, open( experiment_prefix + '/idx_attack.pickle', 'wb'))

# In[7]:


if approximate_meta_gradient:
    gcn_attack = mtk.GNNMetaApprox(_A_obs, _X_obs, labels_self_training, hidden_sizes, 
                                   gpu_id=gpu_id, _lambda=lambda_, train_iters=train_iters, dtype=dtype)
else:
    gcn_attack = mtk.GNNMeta(_A_obs, _X_obs.toarray().astype("float32"), labels_self_training, hidden_sizes, 
                             gpu_id=gpu_id, attack_features=False, train_iters=train_iters, dtype=dtype)


# In[8]:


gcn_attack.build()
gcn_attack.make_loss(ll_constraint=enforce_ll_constrant)


# In[9]:


if approximate_meta_gradient:
    gcn_attack.attack(perturbations, split_train, split_unlabeled, idx_attack)
else:
    gcn_attack.attack(perturbations, split_train, idx_attack)




# In[10]:


adjacency_changes = gcn_attack.adjacency_changes.eval(session=gcn_attack.session).reshape(_A_obs.shape)
modified_adjacency = gcn_attack.modified_adjacency.eval(session=gcn_attack.session)


# In[11]:


re_trainings = 20


# In[12]:


gcn_before_attack = mtk.GCNSparse(sp.csr_matrix(_A_obs), _X_obs, _Z_obs, hidden_sizes, gpu_id=gpu_id)
gcn_before_attack.build(with_relu=True)


# In[13]:


accuracies_clean_unlabeled = []
accuracies_clean_val = []
accuracies_clean_test = []
accuracies_clean_train = []
for _it in tqdm(range(re_trainings)):
    gcn_before_attack.train(split_train, initialize=True, display=False)
    accuracy_clean_val = (gcn_before_attack.logits.eval(session=gcn_before_attack.session).argmax(1) == _z_obs)[split_val].mean()
    accuracies_clean_val.append(accuracy_clean_val)
    accuracy_clean_unlabeled = (gcn_before_attack.logits.eval(session=gcn_before_attack.session).argmax(1) == _z_obs)[split_unlabeled].mean()
    accuracies_clean_unlabeled.append(accuracy_clean_unlabeled)
    accuracy_clean_test = (gcn_before_attack.logits.eval(session=gcn_before_attack.session).argmax(1) == _z_obs)[split_test].mean()
    accuracies_clean_test.append(accuracy_clean_test)
    accuracy_clean_train = (gcn_before_attack.logits.eval(session=gcn_before_attack.session).argmax(1) == _z_obs)[split_train].mean()
    accuracies_clean_train.append(accuracy_clean_train)
    if record_experiment:
        pickle.dump(accuracy_clean_unlabeled, open( experiment_prefix + '/accuracy_clean_unlabeled_'+str(_it)+'.pickle', 'wb'))
        pickle.dump(accuracy_clean_val, open( experiment_prefix + '/accuracy_clean_val_'+str(_it)+'.pickle', 'wb'))
        pickle.dump(accuracy_clean_train, open( experiment_prefix + '/accuracy_clean_train_'+str(_it)+'.pickle', 'wb'))
        pickle.dump(accuracy_clean_test, open( experiment_prefix + '/accuracy_clean_test_'+str(_it)+'.pickle', 'wb'))

# In[14]:


gcn_after_attack = mtk.GCNSparse(sp.csr_matrix(modified_adjacency), _X_obs, _Z_obs, hidden_sizes, gpu_id=gpu_id)
gcn_after_attack.build(with_relu=True)


# In[15]:

accuracies_atk_train = []
accuracies_atk_test = []
accuracies_atk_unlabeled = []
accuracies_atk_val = []
for _it in tqdm(range(re_trainings)):
    gcn_after_attack.train(split_train, initialize=True, display=False)
    accuracy_perturbed_unlabeled = (gcn_after_attack.logits.eval(session=gcn_after_attack.session).argmax(1) == _z_obs)[split_unlabeled].mean()
    accuracies_atk_unlabeled.append(accuracy_perturbed_unlabeled)
    accuracy_perturbed_val = (gcn_after_attack.logits.eval(session=gcn_after_attack.session).argmax(1) == _z_obs)[split_val].mean()
    accuracies_atk_val.append(accuracy_perturbed_val)
    accuracy_perturbed_train = (gcn_after_attack.logits.eval(session=gcn_after_attack.session).argmax(1) == _z_obs)[split_train].mean()
    accuracies_atk_train.append(accuracy_perturbed_train)
    accuracy_perturbed_test = (gcn_after_attack.logits.eval(session=gcn_after_attack.session).argmax(1) == _z_obs)[split_test].mean()
    accuracies_atk_test.append(accuracy_perturbed_test)
    if record_experiment:
        pickle.dump(accuracy_perturbed_unlabeled, open( experiment_prefix + '/accuracy_perturbed_unlabeled_'+str(_it)+'.pickle', 'wb'))
        pickle.dump(accuracy_perturbed_val, open( experiment_prefix + '/accuracy_perturbed_val_' + str(_it) + '.pickle', 'wb'))
        pickle.dump(accuracy_perturbed_train, open( experiment_prefix + '/accuracy_perturbed_train_'+str(_it)+'.pickle', 'wb'))
        pickle.dump(accuracy_perturbed_test, open( experiment_prefix + '/accuracy_perturbed_test_' + str(_it) + '.pickle', 'wb'))

# In[16]:

plt.figure(figsize=(6,6))
chart = sns.boxplot(x=["Orig test-set", "Pert test-set","Orig val-set", "Pert val-set","Orig both", "Pert both"], y=[accuracies_clean_test, accuracies_atk_test, accuracies_clean_val, accuracies_atk_val,accuracies_clean_unlabeled, accuracies_atk_unlabeled])#, re_trainings*[accuracy_logistic]])
plt.title("Accuracy before and after perturbing "+str(int(share_perturbations*100))+"% edges using "+ variant)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.savefig( experiment_prefix + "/plot.png", dpi=600)
plt.show()

plt.figure(figsize=(6,6))
sns.boxplot(x=["Acc. Clean Train", "Acc. Perturbed Train"], y=[ accuracies_clean_train, accuracies_atk_train])#, re_trainings*[accuracy_logistic]])
plt.title("Accuracy before and after perturbing "+str(int(share_perturbations*100))+"% edges using "+ variant)
plt.savefig( experiment_prefix + "/trainplot.png", dpi=600)
plt.show()


if record_experiment:
    pickle.dump(modified_adjacency, open( experiment_prefix + '/modified_adjacency.pickle', 'wb'))
    pickle.dump(adjacency_changes, open( experiment_prefix + '/adjacency_changes.pickle', 'wb'))

