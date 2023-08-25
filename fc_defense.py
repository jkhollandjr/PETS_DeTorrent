import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import bisect
from models import CONV_ATTACK, FC_CONV_EMBEDDER, LSTM_ATTACK

#HYPERPARAMETERS
BATCH_SIZE = 40
PADDING = 3200
FOLD_NUM = int(sys.argv[1])
device = torch.device("cuda")
torch.set_printoptions(sci_mode=False)

train_dir = np.load('fc_data_cv/train_set_' + str(FOLD_NUM) + '.npy')
val_dir = np.load('fc_data_cv/val_set_' + str(FOLD_NUM) + '.npy')
test_dir = np.load('fc_data_cv/test_set_' + str(FOLD_NUM) + '.npy')

train_timing = np.load('fc_data_cv/train_set_timing_' + str(FOLD_NUM) + '.npy')
val_timing = np.load('fc_data_cv/val_set_timing_' + str(FOLD_NUM) + '.npy')
test_timing = np.load('fc_data_cv/test_set_timing_' + str(FOLD_NUM) + '.npy')

train_labels = np.load('fc_data_cv/train_set_labels_' + str(FOLD_NUM) + '.npy')
val_labels = np.load('fc_data_cv/val_set_labels_' + str(FOLD_NUM) + '.npy')
test_labels = np.load('fc_data_cv/test_set_labels_' + str(FOLD_NUM) + '.npy')

embedder = FC_CONV_EMBEDDER().to(device)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=.0001)
embedder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(embedder_optimizer, mode='min', factor=np.sqrt(.1), cooldown=0, min_lr=1e-5)

class TrainData(Dataset):
    def __init__(self, train_dir, train_timing, train_labels, device):
        self.x_dir = train_dir
        self.timing = train_timing
        self.labels = train_labels
        self.device = device

    def __getitem__(self, index):

        if(np.random.randint(2) == 0):
            inflow = torch.from_numpy(self.x_dir[index,:256]).float().to(self.device)
            outflow = torch.from_numpy(self.x_dir[index,256:]).float().to(self.device)
            timing = torch.from_numpy(self.timing[index]).float().to(self.device)
            label = torch.from_numpy(self.labels[index]).float().to(self.device)
            target = torch.ones(1).to(device)

        else:
            r = np.random.randint(12000)
            inflow = torch.from_numpy(self.x_dir[r,:256]).float().to(self.device)
            outflow = torch.from_numpy(self.x_dir[index,256:]).float().to(self.device)
            timing = torch.from_numpy(self.timing[index]).float().to(self.device)
            label = torch.from_numpy(self.labels[index]).float().to(self.device)
            target = torch.zeros(1).to(device)

        return inflow, outflow, timing, label, target

    def __len__(self):
        return len(self.x_dir)

class ValidationData(Dataset):
    def __init__(self, val_dir, val_timing, val_labels, device):
        self.x_dir = val_dir
        self.timing = val_timing
        self.labels = val_labels
        self.device = device

    def __getitem__(self, index):

        if(np.random.randint(2) == 0):
            inflow = torch.from_numpy(self.x_dir[index,:256]).float().to(self.device)
            outflow = torch.from_numpy(self.x_dir[index,256:]).float().to(self.device)
            timing = torch.from_numpy(self.timing[index]).float().to(self.device)
            label = torch.from_numpy(self.labels[index]).float().to(self.device)
            target = torch.ones(1).to(device)

        else:
            r = np.random.randint(4000)
            inflow = torch.from_numpy(self.x_dir[r,:256]).float().to(self.device)
            outflow = torch.from_numpy(self.x_dir[index,256:]).float().to(self.device)
            timing = torch.from_numpy(self.timing[index]).float().to(self.device)
            label = torch.from_numpy(self.labels[index]).float().to(self.device)
            target = torch.zeros(1).to(device)

        return inflow, outflow, timing, label, target

    def __len__(self):
        return len(self.x_dir)

class TestData(Dataset):
    def __init__(self, test_dir, test_timing, test_labels, device):
        self.x_dir = test_dir
        self.timing = test_timing
        self.labels = test_labels
        self.device = device

    def __getitem__(self, index):

        if(np.random.randint(2) == 0):
            inflow = torch.from_numpy(self.x_dir[index,:256]).float().to(self.device)
            outflow = torch.from_numpy(self.x_dir[index,256:]).float().to(self.device)
            timing = torch.from_numpy(self.timing[index]).float().to(self.device)
            label = torch.from_numpy(self.labels[index]).float().to(self.device)
            target = torch.ones(1).to(device)

        else:
            r = np.random.randint(4000)
            inflow = torch.from_numpy(self.x_dir[r,:256]).float().to(self.device)
            outflow = torch.from_numpy(self.x_dir[index,256:]).float().to(self.device)
            timing = torch.from_numpy(self.timing[index]).float().to(self.device)
            label = torch.from_numpy(self.labels[index]).float().to(self.device)
            target = torch.zeros(1).to(device)

        return inflow, outflow, timing, label, target

    def __len__(self):
        return len(self.x_dir)

train_dataset = TrainData(train_dir, train_timing, train_labels, device)
test_dataset = TestData(val_dir, val_timing, val_labels, device)
val_dataset = ValidationData(test_dir, test_timing, test_labels, device)

train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=40, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True)

INPUT_SIZE = 32
HIDDEN_DIM = 128

generator = LSTM_ATTACK(INPUT_SIZE, HIDDEN_DIM).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=.001)

triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1)
loss = nn.BCELoss()

def TrainEpoch(embedder_optimizer, generator_optimizer, counter):
    embedding_loss_list = []
    train_correct = 0
    total = 0
    for batch_idx, (inflow, outflow, timing, label, target) in enumerate(train_loader):
        generator_optimizer.zero_grad()

        inflow = inflow.reshape(40, 1, 256)
        outflow = outflow.reshape(40, 1, 256)
        noise = torch.randn(40, 1, 32).to(device)
        noise = noise.repeat(1, 256, 1)
        timing = torch.div(torch.arange(256), 10)
        timing = timing.repeat(40).reshape(40, 256).to(device) 
        noise[:,:,-1:] = inflow[:,:,:].reshape(40, 256, 1)
        noise[:,:,5] = timing

        inflow_padding = generator(noise).reshape(40, 1, 256)
        inflow_padding = torch.mul(inflow_padding, torch.div(BATCH_SIZE*PADDING, torch.sum(inflow_padding+.0001)))
        inflow_padding = torch.roll(inflow_padding, 1, 2)
        inflow_padding[:,:,0] = np.random.randint(20)

        padded_inflow = torch.add(inflow_padding, inflow)
        padded_inflow = padded_inflow.reshape(40, 256)
 
        output = embedder(padded_inflow, outflow)
        output_loss = loss(output, target)
        generator_loss = -loss(output, target)
        generator_loss.backward()
        generator_optimizer.step()

        embedder_optimizer.zero_grad()

        inflow = inflow.reshape(40, 1, 256)
        outflow = outflow.reshape(40, 1, 256)

        noise = torch.randn(40, 1, 32).to(device)
        noise = noise.repeat(1, 256, 1)
        timing = torch.div(torch.arange(256), 10)
        timing = timing.repeat(40).reshape(40, 256).to(device) 

        noise[:,:,-1:] = inflow[:,:,:].reshape(40, 256, 1)
        noise[:,:,5] = timing
        inflow_padding = generator(noise).reshape(40, 1, 256)
        inflow_padding = torch.mul(inflow_padding, torch.div(BATCH_SIZE*PADDING, torch.sum(inflow_padding+.0001)))
        inflow_padding = torch.roll(inflow_padding, 1, 2)
        inflow_padding[:,:,0] = np.random.randint(20)

        padded_inflow = torch.add(inflow_padding, inflow)
        padded_inflow = padded_inflow.reshape(40, 256)

        output = embedder(padded_inflow, outflow)

        output_loss = loss(output, target)
        output_loss.backward()
        embedder_optimizer.step()
    
        pred = torch.round(output.reshape(40))
        t = target.reshape(40)
        train_correct += pred.eq(t.view_as(pred)).sum().item()
        total += len(pred)

        embedding_loss_list.append(output_loss.item())
    print("Attack Training Accuracy: {}".format(train_correct / float(total)))

  
embedder.eval()

for i in range(60):
    print("\nEPOCH {}".format(i))
    embedder.train()
    TrainEpoch(embedder_optimizer, generator_optimizer, 1)
    embedder.eval()

    val_correct = 0
    total = 0
    for batch_idx, (inflow, outflow, timing, label, target) in enumerate(val_loader):
        inflow = inflow.reshape(40, 1, 256)
        outflow = outflow.reshape(40, 1, 256)
        
        noise = torch.randn(40, 1, 32).to(device)
        noise = noise.repeat(1, 256, 1)
        timing = torch.div(torch.arange(256), 10)
        timing = timing.repeat(40).reshape(40, 256).to(device) 

        noise[:,:,-1:] = inflow[:,:,:].reshape(40, 256, 1)
        noise[:,:,5] = timing
        inflow_padding = generator(noise).reshape(40, 1, 256)
        inflow_padding = torch.mul(inflow_padding, torch.div(BATCH_SIZE*PADDING, torch.sum(inflow_padding+.0001)))
        inflow_padding = torch.roll(inflow_padding, 1, 2)
        inflow_padding[:,:,0] = np.random.randint(20)

        padded_inflow = torch.add(inflow_padding, inflow)
        padded_inflow = padded_inflow.reshape(40, 256)
       
        output = embedder(padded_inflow, outflow)
        pred = torch.round(output.reshape(40))
        t = target.reshape(40)
        val_correct += pred.eq(t.view_as(pred)).sum().item()
        total += len(pred)

    print("Attack Validation Accuracy: {}".format(val_correct / float(total)))

torch.save(generator.state_dict(), 'fc_generator.pth')
torch.save(embedder.state_dict(), 'fc_discriminator.pth')

print("Outputting defended traces")
def rate_estimator(trace):
    average_list = []
    average_list.append(1)
    k = 1
    running_average = 0.0
    for i in range(1, len(trace)):
        running_average = (running_average+k)*math.exp(-k*(trace[i] - trace[i-1]))
        average_list.append(running_average)

    return average_list

OUTPUT = "fc_defense_output/"

for batch_idx, (inflow, outflow, timing, label, target) in enumerate(test_loader):
    train_dir = timing.reshape(40, 10000)
    inflow = inflow.reshape(40, 1, 256)

    timing = torch.div(torch.arange(256), 10)
    timing = timing.repeat(40).reshape(40, 256).to(device)

    noise = torch.randn(40, 1, 32).to(device)
    noise = noise.repeat(1, 256, 1)
    noise[:,:,-1:] = inflow[:,:,:].reshape(40, 256, 1)
    noise[:,:,5] = timing
    inflow_padding = generator(noise).reshape(40, 1, 256)
    inflow_padding = torch.mul(inflow_padding, torch.div(BATCH_SIZE*PADDING, torch.sum(inflow_padding+.0001)))
    inflow_padding = torch.roll(inflow_padding, 1, 2)
    inflow_padding[:,:,0] = np.random.randint(20)

    output_traffic = torch.add(inflow_padding, inflow)
    output_traffic = output_traffic.reshape(40, 1, 256)

    output_traffic = output_traffic.detach().cpu().numpy()
    padding = inflow_padding.detach().cpu().numpy()
    trace_volume = inflow.detach().cpu().numpy()
    trace = train_dir.detach().cpu().numpy()

    label = label.detach().cpu().numpy()

    #add padding based on LSTM strategy and exponential inter-arrival times
    for i in range(BATCH_SIZE):
        trace_rows = []
        base_counter = 0
        timing = np.subtract(np.geomspace(1, 50.0, num=257), 1)
        upload_counter = 0
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
            trace_rows.append((download_intro_time, -1500))
            download_list.append(download_intro_time)
        while(upload_intro_time < tenth_packet):
            upload_intro_time += np.random.exponential(1/10)
            trace_rows.append((upload_intro_time, 500))
            upload_list.append(upload_intro_time)

        #add download packets
        for j in range(255):
            download_padding = padding[i][0][j]
            download_burst = 0.0
            upload_burst = 0.0
            gap_length = (timing[j+1] - timing[j])
            packet_rate = download_padding / float(gap_length)
            upload_packets = download_padding / float(5)

            burst_delay = gap_length*np.random.uniform()

            download_burst = 0.0
            for k in range(int(download_padding)):
                download_burst += np.random.exponential(scale=1/packet_rate)
                time = timing[j] + download_burst + burst_delay + tenth_packet
                trace_rows.append((time, -1500))
                download_list.append(time)
        
        #add upload packets
        download_list.sort()
        upload_list.sort()

        download_rate = rate_estimator(download_list)
        upload_rate = rate_estimator(upload_list)

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
                trace_rows.append((upload_pad_time, 500))
                padding_added += 1
                
            upload_pad_time += .002


        #assuming that MTU-sized packets are being sent (this is done for all evaluated defenses)
        for j in range(10000):
            if(trace[i][j] < 0):
                trace_rows.append((abs(trace[i][j]), -1500))
            elif(trace[i][j] > 0):
                trace_rows.append((abs(trace[i][j]), 500))

        trace_rows = sorted(trace_rows, key=lambda x: x[0])

        with open(OUTPUT + str(int(label[i][0])) + "_" + str(int(label[i][1])), "w") as wf:
            for time in trace_rows:
                rounded_time = str(round(time[0], 4))
                row = "{}\t{}\n".format(rounded_time, time[1])
                wf.write(row)

