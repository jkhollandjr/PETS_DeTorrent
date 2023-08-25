''' Loosely based on VAR-CNN preprocessing code '''
import os
import numpy as np
import random
import json
import math
import sys

np.set_printoptions(threshold=sys.maxsize)

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    Args:
        y (numpy array): class vector to be converted into a matrix
        num_classes (int): total number of classes, if `None` it's inferred from the input array

    Returns:
        numpy array: A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def process_trace(args):
    """Get direction, timing, and label for trace"""
    dir_name = args[0]
    trace_path = args[1]

    with open(os.path.join(dir_name, trace_path), 'r') as f:
        lines = f.readlines()

    dir_seq = np.zeros(10000, dtype=np.int8)
    time_seq = np.zeros(10000, dtype=np.float32)
    label = np.array([int(trace_path.split('_')[0]), int(trace_path.split('_')[1])])

    counter = 0
    for packet_num, line in enumerate(lines):
        line = line.split('\t')
        curr_time = float(line[0])
        curr_dir = np.sign(int(float(line[1].strip())))
        #filtering out small upload/acknowledgement packets
        if(abs(float(line[1].strip())) < 100):
            continue

        if packet_num < 10000:
            dir_seq[counter] = curr_dir
            time_seq[counter] = curr_time
            counter += 1

    return dir_seq, time_seq, label


def main():

    #HYPERPARAMETERS
    num_sites = 20000
    data_dir = 'dcf_dataset/'

    arg_list = []
    for trace_path in os.listdir(str(data_dir+'inflow/')):
        arg_list.append(trace_path)
    arg_list = arg_list[:20000]

    # set up the output
    inflow_idx = 0
    dir_seq_inflow = [None] * num_sites
    time_seq_inflow = [None] * num_sites
    labels_inflow = [None] * num_sites 
    
    outflow_idx = 0
    dir_seq_outflow = [None] * num_sites
    time_seq_outflow = [None] * num_sites
    labels_outflow = [None] * num_sites 


    print('size of total list: %d' % len(arg_list))
    for i in range(len(arg_list)):
        dir_seq, time_seq, label = process_trace([data_dir + 'inflow/', arg_list[i]])
        if i % 10000 == 0:
            print("processed", i)

        dir_seq_inflow[inflow_idx] = dir_seq
        time_seq_inflow[inflow_idx] = time_seq
        labels_inflow[inflow_idx] = label
        inflow_idx += 1

        dir_seq, time_seq, label = process_trace([data_dir + 'outflow/', arg_list[i]])
        dir_seq_outflow[outflow_idx] = dir_seq
        time_seq_outflow[outflow_idx] = time_seq
        labels_outflow[outflow_idx] = label
        outflow_idx += 1

    # save monitored traces
    dir_seq_inflow = np.array(dir_seq_inflow, dtype=np.int8)
    time_seq_inflow = np.array(time_seq_inflow, dtype=np.float32)
    labels_inflow = np.array(labels_inflow)
    
    dir_seq_outflow = np.array(dir_seq_outflow, dtype=np.int8)
    time_seq_outflow = np.array(time_seq_outflow, dtype=np.float32)
    labels_outflow = np.array(labels_outflow)


    #shuffle, split train/val/test, convert from absolute to inter-packet times, add 3rd dim for CNN, one-hot encoding of labels

    #converting to inter-packet times
    abs_time_inflow = time_seq_inflow.copy()
    abs_time_outflow = time_seq_outflow.copy()

    time_seq_inflow[:, 1:] = time_seq_inflow[:, 1:] - time_seq_inflow[:, :-1]
    time_seq_outflow[:, 1:] = time_seq_outflow[:, 1:] - time_seq_outflow[:, :-1]

    #labels_mon = to_categorical(labels_mon, num_classes=num_mon_sites)

    def build_histogram_256(dir_time, abs_time):
        tenth_packet_time = abs_time[10]
        abs_time = abs_time[10:]
        for i in range(len(abs_time)):
            abs_time[i] = abs_time[i] - tenth_packet_time
        dir_time = dir_time[10:]

        timing = np.subtract(np.geomspace(1, 50.0, num=257),1)
        trace_hist = np.zeros(256)
        base_bin = 0
        for i in range(len(abs_time)):
            if(abs_time[i] < 50 and abs_time[i] > 0):
                if(dir_time[i] < 0):
                    for j in range(base_bin, 256):
                        if(abs_time[i] > timing[base_bin] and abs_time[i] < timing[base_bin+1]):
                            hist_bin = base_bin
                            trace_hist[hist_bin] += 1
                            break
                        elif(abs_time[i] > timing[base_bin+1]):
                            base_bin += 1
                            
        trace_hist = trace_hist.reshape((256,1)).astype(int)
        
        return trace_hist

    def get_packet_timing(direction, abs_time):
        upload_packets = []
        for i in range(len(direction)):
            if(direction[i] < 0 and abs_time[i] < 50.0):
                upload_packets.append(-1*abs_time[i])
            if(direction[i] > 0 and abs_time[i] < 50.0):
                upload_packets.append(abs_time[i])

        if(len(upload_packets) > 10000):
            upload_packets = upload_packets[:10000]
        else:
            added_length = 10000 - len(upload_packets)
            for i in range(added_length):
                upload_packets.append(0)

        return np.asarray(upload_packets)

    all_traffic = []
    all_inflow_timing = []
    for i in range(len(time_seq_inflow)):
        if(i%1000==0):
            print(i)
    
        inflow_vol = build_histogram_256(dir_seq_inflow[i], abs_time_inflow[i]).reshape(256)
        outflow_vol = build_histogram_256(dir_seq_inflow[i], abs_time_outflow[i]).reshape(256)
        inflow_timing = get_packet_timing(dir_seq_inflow[i], abs_time_inflow[i])
        traffic = np.concatenate((inflow_vol, outflow_vol), axis=0) 
        all_traffic.append(traffic)
        all_inflow_timing.append(inflow_timing)

    all_traffic = np.array(all_traffic)
    all_inflow_timing = np.array(all_inflow_timing)
    all_labels = labels_inflow

    grouped_data = []
    for volume_data, inflow_timing, label in zip(all_traffic, all_inflow_timing, all_labels):
        grouped_data.append((volume_data, inflow_timing, label))

    random.shuffle(grouped_data)

    num_traces = len(grouped_data)

    FOLDS = 5
    for i in range(FOLDS):
        block_size = int(num_traces / FOLDS)

        test_set = grouped_data[(i%FOLDS)*block_size:((i)%FOLDS)*block_size + block_size]
        val_set = grouped_data[((i+1)%FOLDS)*block_size:((i+1)%FOLDS)*block_size + block_size]
        train_set = grouped_data[((i+2)%FOLDS)*block_size:((i+2)%FOLDS)*block_size + block_size] + grouped_data[((i+3)%FOLDS)*block_size:((i+3)%FOLDS)*block_size + block_size] + grouped_data[((i+4)%FOLDS)*block_size:((i+4)%FOLDS)*block_size + block_size]

        train_hist = np.array([x[0] for x in train_set])
        val_hist = np.array([x[0] for x in val_set])
        test_hist = np.array([x[0] for x in test_set])

        train_timing = np.array([x[1] for x in train_set])
        val_timing = np.array([x[1] for x in val_set])
        test_timing = np.array([x[1] for x in test_set])

        train_labels = np.array([x[2] for x in train_set])
        val_labels = np.array([x[2] for x in val_set])
        test_labels = np.array([x[2] for x in test_set])

        with open('fc_data_cv/train_set_' + str(i) + '.npy', 'wb') as f:
            np.save(f, train_hist)
        with open('fc_data_cv/val_set_' + str(i) + '.npy', 'wb') as f:
            np.save(f, val_hist)
        with open('fc_data_cv/test_set_' + str(i) + '.npy', 'wb') as f:
            np.save(f, test_hist)
        
        with open('fc_data_cv/train_set_timing_' + str(i) + '.npy', 'wb') as f:
            np.save(f, train_timing)
        with open('fc_data_cv/val_set_timing_' + str(i) + '.npy', 'wb') as f:
            np.save(f, val_timing)
        with open('fc_data_cv/test_set_timing_' + str(i) + '.npy', 'wb') as f:
            np.save(f, test_timing)

        with open('fc_data_cv/train_set_labels_' + str(i) + '.npy', 'wb') as f:
            np.save(f, train_labels)
        with open('fc_data_cv/val_set_labels_' + str(i) + '.npy', 'wb') as f:
            np.save(f, val_labels)
        with open('fc_data_cv/test_set_labels_' + str(i) + '.npy', 'wb') as f:
            np.save(f, test_labels)

if __name__ == '__main__':
    main()
