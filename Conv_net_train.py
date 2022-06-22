# libraries
import numpy as np

# classes
from DLD_Utils import DLD_Utils as utl
from Conv_Base import DLD_Net

# parameters 
test_frac = 0.2
epoch = 100
N_EPOCH = 5
batch_size = 64
lr = 0.0002
summary = True

# loading data
dataset_name = "dataset2288"
dataset = utl.load_data(dataset_name)

# Normilizing data and saving the Normilized value        
maxu = np.max(np.abs(dataset[0]), axis=(1, 2), keepdims=True)
maxv = np.max(np.abs(dataset[1]), axis=(1, 2), keepdims=True)
               
# Local MAX
max_vel = np.max((maxu, maxv), axis=0)
max_label = np.max(dataset[2], axis=0)

MAX = []
MAX.append(max_vel)
MAX.append(max_label)

dataset_norm = []
dataset_norm.append(dataset[0]/ MAX[0])
dataset_norm.append(dataset[1]/ MAX[0])
dataset_norm.append(dataset[2]/ MAX[1])

       
# Spiliting data to train test sections
train_ix = np.random.choice(len(dataset_norm[0]), size=int(
    (1-test_frac)*len(dataset_norm[0])), replace=False)

test_ix = np.setdiff1d(np.arange(len(dataset_norm[0])), train_ix)

u_train, v_train, label_train = np.nan_to_num(
    dataset_norm[0][train_ix]),np.nan_to_num(
        dataset_norm[1][train_ix]), np.nan_to_num(
            dataset_norm[2][train_ix])

u_test, v_test, label_test = np.nan_to_num(
    dataset_norm[0][test_ix]),np.nan_to_num(
        dataset_norm[1][test_ix]), np.nan_to_num(
            dataset_norm[2][test_ix])


NN = DLD_Net()
# NN.analyse_data(dataset[0], dataset_norm[0], 3)

label_shape = label_train[0].shape
NN.create_model(label_shape, summary)

# NN.train(u_train, v_train, label_train, u_test, v_test, label_test, epoch, N_EPOCH, batch_size, lr)

NN.DLDNN.load_weights(NN.checkpoint_filepath)

dataset_norm_test = []
dataset_norm_test.append(dataset_norm[0][test_ix])
dataset_norm_test.append(dataset_norm[1][test_ix])
dataset_norm_test.append(dataset_norm[2][test_ix])

#eval_data = NN.network_evaluation(1, dataset_norm_test, MAX)
#import csv
#with open('eval_data.csv', 'w') as file:
#    writer = csv.writer(file)
#    writer.writerows(eval_data)

# label_number = 2227
# f, _, _ = NN.dataset[2][label_number] 
# dp = 0.1
# periods = 1
# start_point = (0, f/2+dp*(1-f)/2)
# NN.strmline_comparison(label_number, dp, periods, start_point)
##############################################################
############### test network for data that it never seen #####
##############################################################

# f = np.round(np.linspace(0.25, 0.75, 10), 2).tolist()
# N = [3.5, 4.5, 5.5, 6.5]
# RE = [0.05, 3, 6, 8, 12, 18]

# loading data
dataset_name = "dataset_test_int"
dataset = utl.load_data(dataset_name)

# Normilizing data and saving the Normilized value        
maxu = np.max(np.abs(dataset[0]), axis=(1, 2), keepdims=True)
maxv = np.max(np.abs(dataset[1]), axis=(1, 2), keepdims=True)
               
# Local MAX
max_vel = np.max((maxu, maxv), axis=0)
# use maximum from the training  data
# max_label = np.max(dataset[2], axis=0)


MAX = []
MAX.append(max_vel)
MAX.append(max_label)

dataset_norm = []
dataset_norm.append(dataset[0]/ MAX[0])
dataset_norm.append(dataset[1]/ MAX[0])
dataset_norm.append(dataset[2]/ MAX[1])

NN.analyse_data(dataset[0], dataset_norm[0], 3)

eval_data = NN.network_evaluation(0.1, dataset_norm, MAX)
import csv
with open('eval_data_test_int.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(eval_data)

# label_number = 51
# f, _, _ = dataset[2][label_number] 
# dp = 0.95
# periods = 1
# start_point = (0, f/2+dp*(1-f)/2)
# NN.strmline_comparison(dataset_norm, MAX, label_number, dp, periods, start_point)
