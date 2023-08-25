import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import math
import bisect
from models import CONV_ATTACK, CUSTOM_CONV_EMBEDDER, LSTM_ATTACK
import sys
import os

BATCH_SIZE=40
DOWNLOAD_PADDING = 3000
BATCH_PADDING = DOWNLOAD_PADDING * BATCH_SIZE
FOLD_NUM = int(sys.argv[1])

device = torch.device("cuda")
torch.set_printoptions(sci_mode=False, edgeitems=10)

#loading data with subpages split
data_dir = 'wf_preprocessed_data/'
train_dir = np.load(data_dir + 'lstm_train_dir_' + str(FOLD_NUM) + '.npy')[:,:10000]
train_hist_512 = np.load(data_dir + 'lstm_train_hist_256_' + str(FOLD_NUM) + '.npy')
train_labels = np.load(data_dir + 'lstm_train_labels_' + str(FOLD_NUM) + '.npy')

val_dir = np.load(data_dir + 'lstm_val_dir_' + str(FOLD_NUM) + '.npy')[:,:10000]
val_hist_512 = np.load(data_dir + 'lstm_val_hist_256_' + str(FOLD_NUM) + '.npy')
val_labels = np.load(data_dir + 'lstm_val_labels_' + str(FOLD_NUM) + '.npy')

test_dir = np.load(data_dir + 'lstm_test_dir_' + str(FOLD_NUM) + '.npy')[:,:10000]
test_hist_512 = np.load(data_dir + 'lstm_test_hist_256_' + str(FOLD_NUM) + '.npy')
test_labels = np.load(data_dir + 'lstm_test_labels_' + str(FOLD_NUM) + '.npy')

train_labels_grouped = np.load(data_dir + 'lstm_train_labels_grouped_' + str(FOLD_NUM) + '.npy')
val_labels_grouped = np.load(data_dir + 'lstm_val_labels_grouped_' + str(FOLD_NUM) + '.npy')
test_labels_grouped = np.load(data_dir + 'lstm_test_labels_grouped_' + str(FOLD_NUM) + '.npy')

embedder = CUSTOM_CONV_EMBEDDER().to(device)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=.0001)
embedder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(embedder_optimizer, mode='min', factor=np.sqrt(.1), cooldown=0, min_lr=1e-6)

discriminator = CUSTOM_CONV_EMBEDDER().to(device)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.0001)

#loading data for basic WF attack
class AttackData(Dataset):
    def __init__(self, train_dir, train_hist_512, train_labels, device):
        self.x_dir = train_dir
        self.x_hist_512 = train_hist_512
        self.y_val = train_labels
        self.device = device

    def __getitem__(self, index):
        x_dir_cuda = torch.from_numpy(self.x_dir[index]).reshape(10000, 1).float().to(self.device)
        x_hist_512_cuda = torch.from_numpy(self.x_hist_512[index]).reshape(256, 1).float().to(self.device)
        y_val_cuda = torch.from_numpy(self.y_val[index]).to(self.device)

        return x_dir_cuda, x_hist_512_cuda, y_val_cuda

    def __len__(self):
        return len(self.x_dir)

class UnmonData(Dataset):
    def __init__(self, unmon_dir, unmon_hist_512, device):
        self.x_dir = unmon_dir
        self.x_hist_512 = unmon_hist_512
        self.device = device

    def __getitem__(self, index):
        x_dir_cuda = torch.from_numpy(self.x_dir[index]).reshape(10000, 1).float().to(self.device)
        x_hist_512_cuda = torch.from_numpy(self.x_hist_512[index]).reshape(256, 1).float().to(self.device)

        return x_dir_cuda, x_hist_512_cuda

    def __len__(self):
        return len(self.x_dir)


#loading data for training embedder using triplet loss
class ValidationData(Dataset):
    def __init__(self, val_hist_512, val_labels, device):
        self.val_hist = val_hist_512
        self.labels = np.argmax(val_labels, axis=1)
        self.device = device

    def __getitem__(self, index):
        index = self.labels[index]
        index_matching = np.where(self.labels == index)[0]
        index_nonmatching = np.where(self.labels != index)[0]

        index_a, index_p = np.random.choice(index_matching, 2, replace=True)
        index_n = np.random.choice(index_nonmatching, 1)[0]

        hist_a = self.val_hist[index_a].reshape(256)
        hist_p = self.val_hist[index_p].reshape(256)
        hist_n = self.val_hist[index_n].reshape(256)

        hist_a = torch.from_numpy(hist_a).float().to(self.device)
        hist_p = torch.from_numpy(hist_p).float().to(self.device)
        hist_n = torch.from_numpy(hist_n).float().to(self.device)

        return hist_a, hist_p, hist_n
    
    def __len__(self):
        return len(self.labels)

class TestData(Dataset):
    def __init__(self, test_hist_512, test_labels, device):
        self.test_hist = test_hist_512
        self.labels = np.argmax(test_labels, axis=1)
        self.device = device

    def __getitem__(self, index):
        index = self.labels[index]
        index_matching = np.where(self.labels == index)[0]
        index_nonmatching = np.where(self.labels != index)[0]

        index_a, index_p = np.random.choice(index_matching, 2, replace=True)
        index_n = np.random.choice(index_nonmatching, 1)[0]

        hist_a = self.test_hist[index_a].reshape(256)
        hist_p = self.test_hist[index_p].reshape(256)
        hist_n = self.test_hist[index_n].reshape(256)

        hist_a = torch.from_numpy(hist_a).float().to(self.device)
        hist_p = torch.from_numpy(hist_p).float().to(self.device)
        hist_n = torch.from_numpy(hist_n).float().to(self.device)

        return hist_a, hist_p, hist_n

    def __len__(self):
        return len(self.labels)

class TrainData(Dataset):
    def __init__(self, train_hist_512, train_labels, device):
        self.train_hist = train_hist_512
        self.labels = np.argmax(train_labels, axis=1)
        self.device = device

    def __getitem__(self, index):
        index = self.labels[index]

        index_matching = np.where(self.labels == index)[0]
        index_nonmatching = np.where(self.labels != index)[0]

        index_a, index_p = np.random.choice(index_matching, 2, replace=False)
        index_n = np.random.choice(index_nonmatching, 1)[0]

        hist_a = self.train_hist[index_a].reshape(256)
        hist_p = self.train_hist[index_p].reshape(256)
        hist_n = self.train_hist[index_n].reshape(256)

        hist_a = torch.from_numpy(hist_a).float().to(self.device)
        hist_p = torch.from_numpy(hist_p).float().to(self.device)
        hist_n = torch.from_numpy(hist_n).float().to(self.device)
        return hist_a, hist_p, hist_n


    def __len__(self):
        return len(self.train_hist)

train_dataset = TrainData(train_hist_512, train_labels, device)
test_dataset = TestData(test_hist_512, test_labels, device)
val_dataset = ValidationData(val_hist_512, val_labels, device)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_attack = AttackData(train_dir, train_hist_512, train_labels_grouped, device)
train_attack_loader = DataLoader(train_attack, batch_size=BATCH_SIZE, shuffle=True)

test_attack = AttackData(test_dir, test_hist_512, test_labels_grouped, device)
test_attack_loader = DataLoader(test_attack, batch_size=BATCH_SIZE, shuffle=True)

pdist = nn.PairwiseDistance(p=2)

INPUT_SIZE = 32
HIDDEN_DIM = 128
generator = LSTM_ATTACK(INPUT_SIZE, HIDDEN_DIM).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=.0001)

triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1)

attack = CONV_ATTACK().to(device)
attack_optimizer = torch.optim.Adam(attack.parameters(), lr=.001)
attack_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(attack_optimizer, mode='max', factor=np.sqrt(.1), cooldown=0, min_lr=1e-5)

#Training embedder using triplet loss
def TrainEpoch(embedder_optimizer, counter):
    embedding_loss_list = []
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        embedder_optimizer.zero_grad()

        anchor = embedder(anchor.reshape(BATCH_SIZE, 1, 256))
        positive = embedder(positive.reshape(BATCH_SIZE, 1, 256))
        negative = embedder(negative.reshape(BATCH_SIZE, 1, 256))
 
        loss = triplet_loss(anchor, positive, negative)
 
        embedding_loss_list.append(loss.item())

        loss.backward()
        embedder_optimizer.step()
    print("Embedder Training Loss: {}".format(sum(embedding_loss_list)/float(len(embedding_loss_list))))

best_loss = None
for i in range(300):
    print("EPOCH: {}".format(i))
    embedder.train()
    TrainEpoch(embedder_optimizer, i)
    embedder.eval()

    embedding_loss_list = []
    for batch_idx, (anchor, positive, negative) in enumerate(val_loader):
        if(anchor.shape[0]!=BATCH_SIZE):
            continue
        anchor = embedder(anchor.reshape(BATCH_SIZE, 1, 256))
        positive = embedder(positive.reshape(BATCH_SIZE, 1, 256))
        negative = embedder(negative.reshape(BATCH_SIZE, 1, 256))
        loss = triplet_loss(anchor, positive, negative)
        
        embedding_loss_list.append(loss.item())

    average_loss = sum(embedding_loss_list) / len(embedding_loss_list)
    if(not best_loss or average_loss < best_loss):
        best_loss = average_loss
        torch.save(embedder.state_dict(), 'be_embedder_best.pth')
        print('Saving best model')
    print("Embedder Val Loss: {}\n".format(sum(embedding_loss_list)/float(len(embedding_loss_list))))
    
    embedder_scheduler.step(sum(embedding_loss_list)/float(len(embedding_loss_list)))
print("BEST LOSS: {}".format(best_loss))

embedder.load_state_dict(torch.load('be_embedder_best.pth'))

#compute loss on test set
test_loss = []
for batch_idx, (anchor, positive, negative) in enumerate(test_loader):
    if(anchor.shape[0] != BATCH_SIZE):
        continue
    anchor = embedder(anchor.reshape(BATCH_SIZE, 1, 256))
    positive = embedder(positive.reshape(BATCH_SIZE, 1, 256))
    negative = embedder(negative.reshape(BATCH_SIZE, 1, 256))
    
    loss = triplet_loss(anchor, positive, negative)

    test_loss.append(loss.item())

average_loss = sum(test_loss) / len(test_loss)
print("TEST LOSS: {}".format(average_loss))

embedder.eval()

def AdversarialEpoch(embedder_optimizer, discriminator_optimizer, generator_optimizer,  counter):
    embedding_loss_list = []
    generator_loss_list = []
    discriminator_loss_list = []
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor = anchor.reshape(BATCH_SIZE, 1, 256)
        positive = positive.reshape(BATCH_SIZE, 1, 256)
        negative = negative.reshape(BATCH_SIZE, 1, 256)

        #DISCRIMINATOR TRAINING

        #create timing tensor
        discriminator_optimizer.zero_grad()
        timing = torch.div(torch.arange(256), 10)
        timing = timing.repeat(BATCH_SIZE).reshape(BATCH_SIZE, 256).to(device)

        noise = torch.randn(BATCH_SIZE, 1, 32).to(device)
        noise = noise.repeat(1, 256, 1)
        anchor = torch.transpose(anchor, 1, 2)
        #feed noise, timing, and trace data into LSTM
        noise[:,:,-1:] = anchor[:,:,:]
        noise[:,:,5] = timing

        anchor_padding = generator(noise)
        anchor = torch.transpose(anchor, 1, 2)
        anchor_padding = torch.transpose(anchor_padding, 1, 2)
        anchor_padding = torch.mul(anchor_padding, torch.div(BATCH_PADDING, torch.sum(anchor_padding+.0001)))
        
        #apply output of generator to next time step
        anchor_padding = torch.roll(anchor_padding, 1, 2)
        anchor_padding[:,:,0] = np.random.randint(20)
        anchor_padded = torch.add(anchor_padding, anchor)
        anchor_emb = embedder(anchor)

        discriminator_loss = torch.mean(pdist(discriminator(anchor_padded), anchor_emb))
        discriminator_loss_list.append(discriminator_loss.item())

        discriminator_loss.backward()
        discriminator_optimizer.step()

        #START GENERATOR TRAINING
        generator_optimizer.zero_grad()
        #create timing tensor
        timing = torch.div(torch.arange(256), 10)
        timing = timing.repeat(BATCH_SIZE).reshape(BATCH_SIZE, 256).to(device)

        noise = torch.randn(BATCH_SIZE, 1, 32).to(device)
        noise = noise.repeat(1, 256, 1)
        anchor = torch.transpose(anchor, 1, 2)
        #feed noise, timing, and trace data into LSTM
        noise[:,:,-1:] = anchor[:,:,:]
        noise[:,:,5] = timing

        anchor_padding = generator(noise)
        anchor = torch.transpose(anchor, 1, 2)
        anchor_padding = torch.transpose(anchor_padding, 1, 2)
        anchor_padding = torch.mul(anchor_padding, torch.div(BATCH_PADDING, torch.sum(anchor_padding+.0001)))
        #apply output of generator to next time step
        anchor_padding = torch.roll(anchor_padding, 1, 2)
        anchor_padding[:,:,0] = np.random.randint(20)

        anchor_padded = torch.add(anchor_padding, anchor)
        anchor_emb = embedder(anchor)

        #discriminator loss is based on how close its guess of the embedding is to the actual
        guess = discriminator(anchor_padded)
        generator_loss = -torch.mean(pdist(guess, anchor_emb))
        generator_loss_list.append(generator_loss.item())

        generator_loss.backward()
        generator_optimizer.step()

    print("Generator Training Loss: {}".format(sum(generator_loss_list)/float(len(generator_loss_list))))
    print("Discriminator Training Loss: {}".format(sum(discriminator_loss_list)/float(len(discriminator_loss_list))))

#Playing adversarial game between generator and discriminator
for i in range(90):
    print("EPOCH: {}".format(i))
    generator.train()
    discriminator.train()
    AdversarialEpoch(embedder_optimizer, discriminator_optimizer, generator_optimizer, i)
    embedder.eval()
    generator.eval()
    discriminator.eval()

torch.save(generator.state_dict(), 'wf_generator.pth')
torch.save(discriminator.state_dict(), 'wf_discriminator.pth')

#Training basic WF attack to test strength of generated traces
def TrainEpoch(attack_optimizer):
    train_correct = 0
    total = 0
    for batch_idx, (train_dir, train_hist_512, target) in enumerate(train_attack_loader):
        attack_optimizer.zero_grad()
        anchor = train_hist_512.reshape(BATCH_SIZE, 1, 256)

        timing = torch.div(torch.arange(256), 10)
        timing = timing.repeat(BATCH_SIZE).reshape(BATCH_SIZE, 256).to(device)

        noise = torch.randn(BATCH_SIZE, 1, 32).to(device)
        noise = noise.repeat(1, 256, 1)
        anchor = torch.transpose(anchor, 1, 2)
        noise[:,:,-1:] = anchor[:,:,:]
        noise[:,:,5] = timing

        anchor_padding = generator(noise).detach()
        anchor = torch.transpose(anchor, 1, 2)
        anchor_padding = torch.transpose(anchor_padding, 1, 2)
        anchor_padding = torch.round(torch.mul(anchor_padding, torch.div(BATCH_PADDING, torch.sum(anchor_padding+.0001))))
        anchor_padding = torch.roll(anchor_padding, 1, 2)
        anchor_padding[:,:,0] = np.random.randint(20)
        output_traffic = torch.add(anchor_padding, anchor).reshape(BATCH_SIZE, 1, 256)

        output = attack(output_traffic.reshape(BATCH_SIZE, 1, 256))

        pred = output.argmax(dim=1, keepdim=True)
        t = target.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(t.view_as(pred)).sum().item()
        total += len(pred)

        loss = F.cross_entropy(output, t.reshape(BATCH_SIZE))
        loss.backward()
        attack_optimizer.step()
    print("Train Accuracy: {}".format(float(train_correct)/total))

def rate_estimator(trace):
    average_list = []
    average_list.append(1)
    k = 1
    running_average = 0.0
    for i in range(1, len(trace)):
        running_average = (running_average+k)*math.exp(-k*(trace[i] - trace[i-1]))
        average_list.append(running_average)

    return average_list
 

print("Outputting defended traces")

OUTPUT = "wf_defense_output/"
class_counter = [0]*95
if(FOLD_NUM != 0):
    #get number of files of each class in directory
    file_list = [x for x in os.listdir(OUTPUT)]
    class_numbers = [x.split('-')[0] for x in file_list]
    for cn in class_numbers:
        class_counter[int(cn)] += 1
    
for batch_idx, (train_dir, train_hist_512, train_labels) in enumerate(test_attack_loader):
    train_dir = train_dir.reshape(BATCH_SIZE, 10000)
    anchor = train_hist_512.reshape(BATCH_SIZE, 1, 256)

    timing = torch.div(torch.arange(256), 10)
    timing = timing.repeat(BATCH_SIZE).reshape(BATCH_SIZE, 256).to(device)

    noise = torch.randn(BATCH_SIZE, 1, 32).to(device)
    noise = noise.repeat(1, 256, 1)
    anchor = torch.transpose(anchor, 1, 2)
    noise[:,:,-1:] = anchor[:,:,:]
    noise[:,:,5] = timing

    anchor_padding = generator(noise).detach()
    anchor = torch.transpose(anchor, 1, 2)
    anchor_padding = torch.transpose(anchor_padding, 1, 2)
    anchor_padding = torch.round(torch.mul(anchor_padding, torch.div(BATCH_PADDING, torch.sum(anchor_padding+.0001))))

    anchor_padding = torch.roll(anchor_padding, 1, 2)
    anchor_padding[:,:,0] = np.random.randint(20)
    output_traffic = torch.add(anchor_padding, anchor).reshape(BATCH_SIZE, 1, 256)

    output_traffic = output_traffic.detach().cpu().numpy()
    padding = anchor_padding.detach().cpu().numpy()
    trace_volume = anchor.detach().cpu().numpy()
    trace = train_dir.detach().cpu().numpy()

    target = train_labels.detach().cpu().numpy()

    #add padding based on LSTM strategy and exponential inter-arrival times
    for i in range(BATCH_SIZE):
        t = int(np.argmax(target[i]))
        trace_rows = []
        base_counter = 0
        timing = np.subtract(np.geomspace(1, 50, num=257), 1)
        tenth_packet = abs(trace[i][10])

        #get upload and download traces separately
        upload = np.zeros(10000)
        download = np.zeros(10000)
        upload_counter = 0
        download_counter = 0
        for j in range(10000):
            if(trace[i][j] > 0):
                upload[upload_counter] = trace[i][j]
                upload_counter += 1
            else:
                download[download_counter] = trace[i][j]
                download_counter += 1
        
        download_list = [abs(x) for x in download if x<0]
        upload_list = [abs(x) for x in upload if x>0]
 
        #Defend until tenth packet
        download_intro_time = 0.0
        upload_intro_time = 0.0
        while(download_intro_time < tenth_packet):
            download_intro_time += np.random.exponential(1/10)
            trace_rows.append((download_intro_time, -1))
            download_list.append(download_intro_time)
        while(upload_intro_time < tenth_packet):
            upload_intro_time += np.random.exponential(1/10)
            trace_rows.append((upload_intro_time, 1))
            upload_list.append(upload_intro_time)

        #add download packets
        download_added = 0
        for j in range(256):
            download_padding = padding[i][0][j]

            base_time = timing[j]
            end_time = timing[j+1]
            burst_delay = (end_time - base_time)*np.random.uniform()
            upload_burst_delay = (end_time - base_time)*np.random.uniform()

            download_burst = 0.0
            upload_burst = 0.0
            for k in range(int(download_padding)):
                download_added += 1
                rate = download_padding / (end_time - base_time)
                download_burst += np.random.exponential(scale=1/rate)
                time = tenth_packet + base_time + download_burst + burst_delay
                #time = base_time + download_burst + burst_delay
                trace_rows.append((time, -1))
                download_list.append(time)
        
        download_list.sort()
        upload_list.sort()
 
        download_rate = rate_estimator(download_list)
        upload_rate = rate_estimator(upload_list)

        #add upload packets
        #idea: calculate estimated average sending rate for download and upload, then pad upload to goal upload
        k = 1
        upload_pad_time = tenth_packet
        target_rate = 0
        padding_added = 0
        padding_ratio = 5
        while(upload_pad_time < 50):
            #get download rate
            download_index = bisect.bisect(download_list, upload_pad_time) - 1
            if(download_index < 0):
                current_download_rate = 1
            else:
                previous_download_rate = download_rate[download_index]
                gap = upload_pad_time - download_list[download_index]
                current_download_rate = previous_download_rate * math.exp(-k*gap)

            #get upload_rate
            upload_index = bisect.bisect(upload_list, upload_pad_time) - 1
            if(upload_index < 0):
                current_upload_rate = 1
            else:
                previous_upload_rate = upload_rate[upload_index]
                gap = upload_pad_time - upload_list[upload_index]
                current_upload_rate = previous_upload_rate * math.exp(-k*gap)

            if(current_download_rate > current_upload_rate):
                target_rate = current_download_rate / padding_ratio
            else:
                target_rate = 1

            odds = 0
            if(current_upload_rate < target_rate):
                odds = (target_rate - current_upload_rate) / 500

            if(np.random.uniform() < odds):
                trace_rows.append((upload_pad_time, 1))
                padding_added += 1
                
            upload_pad_time += .002

        #add real packets
        for j in range(10000):
            if(trace[i][j] < 0):
                trace_rows.append((abs(trace[i][j]), -1))
            elif(trace[i][j] > 0):
                trace_rows.append((abs(trace[i][j]), 1))

        label = class_counter[t]
        trace_rows = sorted(trace_rows, key=lambda x: x[0])

        with open(OUTPUT + "/" + str(t) + "-" + str(label), "w") as wf:
            for time in trace_rows:
                rounded_time = str(round(time[0], 2))
                row = "{}\t{}\n".format(rounded_time, time[1])
                wf.write(row)
        class_counter[t] += 1

