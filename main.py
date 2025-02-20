import torch.nn as nn
import torch
from new_data.training_data import dataset_maker
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from models.R2Plus1D import R2Plus1DClassifier

import numpy as np
from torch.nn.parallel import DistributedDataParallel
import os
import torchvision.models.video as models
from sklearn.metrics import classification_report
from torchvision.transforms import RandomVerticalFlip, Normalize
import warnings
import random
from sklearn.metrics import confusion_matrix

seed = 4048

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



warnings.filterwarnings("ignore")
fall_dataset_dict = {
    "DAPD_1_25": "/data/user/DAPD_1_25/home/user1/fall",
    "DAPD_1_5": "/data/user/DAPD_1_5/home/user1/fall",
    "DAPD_1_125": "/data/user/DAPD_1_125/home/user1/fall",
    "DAPD_075_25": "/data/user/DAPD_075_25/home/user1/fall",
    "DAPD_125_25": "/data/user/DAPD_125_25/home/user1/fall",
}

non_fall_dataset_dict = {
    "DAPD_1_25": "/data/user/DAPD_1_25/home/user1/non_fall",
    "DAPD_1_5": "/data/user/DAPD_1_5/home/user1/non_fall",
    "DAPD_1_125": "/data/user/DAPD_1_125/home/user1/non_fall",
    "DAPD_075_25": "/data/user/DAPD_075_25/home/user1/non_fall",
    "DAPD_125_25": "/data/user/DAPD_125_25/home/user1/non_fall",
}

test_fall_dataset_dict = {
    "l_u1_fall": "/data/user/DAPD_1_25/lab/user1/fall",
    "l_u2_fall": "/data/user/DAPD_1_25/lab/user2/fall",
    "l_u3_fall": "/data/user/DAPD_1_25/lab/user3/fall",
    "l_u4_fall": "/data/user/DAPD_1_25/lab/user4/fall",
    "l_u5_fall": "/data/user/DAPD_1_25/lab/user5/fall"
}

test_non_fall_dataset_dict = {
    "l_u1_non_fall": "/data/user/DAPD_1_25/lab/user1/non_fall",
    "l_u2_non_fall": "/data/user/DAPD_1_25/lab/user2/non_fall",
    "l_u3_non_fall": "/data/user/DAPD_1_25/lab/user3/non_fall",
    "l_u4_non_fall": "/data/user/DAPD_1_25/lab/user4/non_fall",
    "l_u5_non_fall": "/data/user/DAPD_1_25/lab/user5/non_fall"
}


def read_npy(path, data_label=None):
    amp_list = []
    file = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
    for file_name in file:
        amp = np.load(file_name)
        amp_list.append(amp)
    # np.transpose(np.array(amp_list), axes=(0, 2, 1, 3, 4))
    if data_label == 'Fall':
        return  np.transpose(np.array(amp_list), axes=(0, 2, 1, 3, 4))
    else:
        return  np.transpose(np.concatenate(amp_list, axis=0), axes=(0, 2, 1, 3, 4))

flag = 'DAPD_1_25'
data_label = 'Fall'
if flag == 'DAPD_1_25':
    training_fall = read_npy(fall_dataset_dict["DAPD_125_25"], data_label)
    training_non_fall = read_npy(non_fall_dataset_dict["DAPD_125_25"])

    test_l_u1_fall = read_npy(test_fall_dataset_dict["l_u1_fall"], data_label)

    test_l_u2_fall = read_npy(test_fall_dataset_dict["l_u2_fall"], data_label)

    test_l_u3_fall = read_npy(test_fall_dataset_dict["l_u3_fall"], data_label)

    test_l_u4_fall = read_npy(test_fall_dataset_dict["l_u4_fall"], data_label)

    test_l_u5_fall = read_npy(test_fall_dataset_dict["l_u5_fall"], data_label)

walk_path = "/data/user/user_walk.npy"
stand_path = "/data/user/user_stand.npy"
run_path = "/data/user/user_run.npy"
jump_path = "/data/user/user_jump.npy"

walk_data = np.transpose(np.load(walk_path), axes=(0, 2, 1, 4, 3))
stand_data = np.transpose(np.load(stand_path), axes=(0, 2, 1, 4, 3))
run_data = np.transpose(np.load(run_path), axes=(0, 2, 1, 4, 3))
jump_data = np.transpose(np.load(jump_path), axes=(0, 2, 1, 4, 3))

n1 = 80
n2 = 160

u1_nonfall = np.concatenate((walk_data[int(0*n2):int(n2)], stand_data[int(0*n1):int(n1)], run_data[int(0*n1):int(n1)], jump_data[int(0*n1):int(n1)]), axis=0)
u2_nonfall = np.concatenate((walk_data[int(2*n2):int(3*n2)], stand_data[int(2*n1):int(3*n1)], run_data[int(2*n1):int(3*n1)], jump_data[int(2*n1):int(3*n1)]), axis=0)
u3_nonfall = np.concatenate((walk_data[int(3*n2):int(4*n2)], stand_data[int(3*n1):int(4*n1)], run_data[int(3*n1):int(4*n1)], jump_data[int(3*n1):int(4*n1)]), axis=0)
u4_nonfall = np.concatenate((walk_data[int(4*n2):int(5*n2)], stand_data[int(4*n1):int(5*n1)], run_data[int(4*n1):int(5*n1)], jump_data[int(4*n1):int(5*n1)]), axis=0)
u5_nonfall = np.concatenate((walk_data[int(5*n2):int(6*n2)], stand_data[int(5*n1):int(6*n1)], run_data[int(5*n1):int(6*n1)], jump_data[int(5*n1):int(6*n1)]), axis=0)

print(training_fall.shape)
print(training_non_fall.shape)
print(test_l_u1_fall.shape)
print(test_l_u2_fall.shape)
print(test_l_u3_fall.shape)
print(test_l_u4_fall.shape)
print(test_l_u5_fall.shape)
print(u1_nonfall.shape)
print(u2_nonfall.shape)
print(u3_nonfall.shape)
print(u4_nonfall.shape)
print(u5_nonfall.shape)

# event for fall 1 025
# original_fall = training_fall[:,:,1:-1,:,:]
# aug1_inter1 = training_fall[:,:,0:8,:,:]
# aug1_inter2 = training_fall[:,:,-8:,:,:]

# data augmentation
original_fall = training_fall[:,:,1:-2,:,:]
print(original_fall.shape)
aug1_inter1 = training_fall[:,:,0:8,:,:]
aug1_inter2 = training_fall[:,:,-9:-1,:,:]
new_training1 = np.concatenate((original_fall, aug1_inter1, aug1_inter2), axis=0)
new_training2 = new_training1[:, :, :, ::-1, :]
new_training1 = np.concatenate((new_training1, new_training2), axis=0)
print(original_fall.shape)
print(aug1_inter1.shape)
print(aug1_inter2.shape)
print(new_training1.shape)
# print(aug2.shape)


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            10: 4e-5, 20: 1e-5, 30: 5e-6, 40: 1e-6,
            50: 5e-7, 60: 1e-7, 70: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def metrics_(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    # 计算指标
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
    print('CM:', cm)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, FPR: {fpr:.4f}')
    return accuracy

def other_env_evl(fall_loader, model, criterion, env):
    model.eval()
    test_loss_fall = 0.0
    correct_fall = 0
    total_fall = 0
    true_label = []
    predicted_label = []

    with torch.no_grad():
        for inputs, labels in fall_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss_fall += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_fall += labels.size(0)
            correct_fall += (predicted == labels).sum().item()

            true_label += labels.cpu().numpy().tolist()
            predicted_label += predicted.cpu().numpy().tolist()
    print(env, ': ---------------------------------------')
    acc = metrics_(true_label, predicted_label)
    if env == 'lab':
        return acc

    


def train_model(train_loader, test_loader, model, num_epochs, learning_rate, patience, device,
                env1_lodaer, env2_lodaer, env3_lodaer, env4_lodaer, env5_lodaer, env6_lodaer, best_model_param_path=None, param_load=None):
    model.to(device)
    if param_load:
        model.load_state_dict(torch.load(best_model_param_path))
    earlystop_path = './model_parameter'
    train_loss, train_acc, test_acc = [], [], []
    model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    test_accuracy_list = []
    report_list = []
    for epoch in range(num_epochs):
        if epoch+1 > 16:
            for param in model.parameters():
                param.requires_grad = True
                
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total

        # 测试阶段\
        print(f'Epoch {epoch}: ==============================================================================')
        acvc = other_env_evl(test_loader, model, criterion, env='lab')
        test_accuracy_list.append(acvc)
        other_env_evl(env1_lodaer, model, criterion, env='l1')
        other_env_evl(env2_lodaer, model, criterion, env='l2')
        other_env_evl(env3_lodaer, model, criterion, env='l3')
        other_env_evl(env4_lodaer, model, criterion, env='l4')
        other_env_evl(env5_lodaer, model, criterion, env='l5')
    return test_accuracy_list, report_list


label = [0 for i in range(new_training1.shape[0])] + [1 for i in range(training_non_fall.shape[0])]
train_data = np.concatenate((new_training1, training_non_fall), axis=0)
X_train, X_test, y_train, y_test = train_test_split(
        train_data, label, test_size=0.2, stratify=label, random_state=seed
    )
X_train, X_test = torch.tensor(np.array(X_train), dtype=torch.float32), torch.tensor(np.array(X_test), dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def test_data(fall, non_fall):
    dataset = np.concatenate((fall, non_fall), axis=0)
    dataset = torch.tensor(dataset, dtype=torch.float32)

    label = [0 for i in range(len(fall))] + [1 for i in range(len(non_fall))]
    label = torch.tensor(label, dtype=torch.long)

    non_fall_dataset = TensorDataset(dataset, label)
    loader = DataLoader(non_fall_dataset, batch_size=128, shuffle=False)
    return loader


u1_loader = test_data(test_l_u1_fall, u1_nonfall)
u2_loader = test_data(test_l_u2_fall, u2_nonfall)
u3_loader = test_data(test_l_u3_fall, u3_nonfall)
u4_loader = test_data(test_l_u4_fall, u4_nonfall)
u5_loader = test_data(test_l_u5_fall, u5_nonfall)

    


# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.r2plus1d_18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
# model = nn.DataParallel(model)


for param in model.parameters():
    param.requires_grad = False
    
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters(): 
    param.requires_grad = True


    
def input_data2(data_list, flag=None):
    labels = [0, 1]
    num_workers = 4 
    test_data = []
#other, fall
    test_data = torch.tensor(data_list[0].transpose(0,2,1,3,4))
    print(test_data.shape)
    test_labels = []
    if flag == 'fall':
        # train_labels.extend([label] * len(train_data[i]))
        test_labels.extend([0] * test_data.shape[0])
    else:
        test_labels.extend([1] * test_data.shape[0])
        

    test_dataset = TensorDataset(test_data, torch.tensor(test_labels))

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=num_workers)
    return test_loader



notpre_test_accuracy_list, notpre_report_list = train_model(train_loader, test_loader, model, num_epochs=100, learning_rate=0.01, patience=10, device=device, env1_lodaer=room_loader, env2_lodaer=u1_loader, env3_lodaer=u2_loader, env4_lodaer=u3_loader, env5_lodaer=u4_loader, env6_lodaer=u5_loader, best_model_param_path='p', param_load=None)
