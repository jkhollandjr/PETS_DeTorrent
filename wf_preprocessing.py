'''Loosely based on Var-CNN preprocessing'''
import numpy as np
import os
import random
import math

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


def get_trace(args):
    '''Get timing and direction information from trace'''
    dir_name = args[0]
    trace_path = args[1]

    with open(os.path.join(dir_name, trace_path), 'r') as f:
        lines = f.readlines()

    dir_seq = np.zeros(10000, dtype=np.int8)
    time_seq = np.zeros(10000, dtype=np.float32)

    if('-' not in trace_path):
        label = 0
    else:
        label = int(trace_path.split('-')[0]) + 1

    for packet_num, line in enumerate(lines):
        line = line.split('\t')
        curr_time = float(line[0])
        curr_dir = np.sign(int(float(line[1].strip())))

        if packet_num < 10000:
            dir_seq[packet_num] = curr_dir
            time_seq[packet_num] = curr_time

    return dir_seq, time_seq, label


def main():

    #HYPERPARAMERS
    num_mon_sites = 950
    num_mon_inst = 20
    num_unmon_sites = 19000
    data_dir = 'be_dataset/'
    output_dir = 'wf_preprocessed_data/'

    arg_list = []
    for trace_path in os.listdir(data_dir):
        arg_list.append([data_dir, trace_path])

    # set up the output
    mon_idx = 0
    unmon_idx = 0
    dir_seq_mon = [None] * (num_mon_sites * num_mon_inst)
    time_seq_mon = [None] * (num_mon_sites * num_mon_inst)
    labels_mon = [None] * (num_mon_sites * num_mon_inst)

    dir_seq_unmon = [None] * (num_unmon_sites)
    time_seq_unmon = [None] * (num_unmon_sites)

    print('size of total list: %d' % len(arg_list))
    for i in range(len(arg_list)):
        dir_seq, time_seq, label = get_trace(arg_list[i])

        if i % 10000 == 0:
            print("processed", i)

        if label == 0:  # unmon site
            dir_seq_unmon[unmon_idx] = dir_seq
            time_seq_unmon[unmon_idx] = time_seq
            #labels_unmon[unmon_idx] = label
            unmon_idx += 1
        else:
            if(len(dir_seq) != 10000):
                print(len(dir_seq))
            dir_seq_mon[mon_idx] = dir_seq
            time_seq_mon[mon_idx] = time_seq
            labels_mon[mon_idx] = label
            mon_idx += 1

    print(len(dir_seq_unmon))

    # save monitored traces
    dir_seq_mon = np.array(dir_seq_mon, dtype=np.int8)
    time_seq_mon = np.array(time_seq_mon, dtype=np.float32)
    labels_mon = np.array(labels_mon)

    # save unmonitored traces
    dir_seq_unmon = np.array(dir_seq_unmon, dtype=np.int8)
    time_seq_unmon = np.array(time_seq_unmon, dtype=np.float32)

    #converting to inter-packet times
    abs_time_mon = time_seq_mon.copy()
    time_seq_mon[:, 1:] = time_seq_mon[:, 1:] - time_seq_mon[:, :-1]

    if(num_unmon_sites > 0):
        abs_time_unmon = time_seq_unmon.copy()
        time_seq_unmon[:, 1:] = time_seq_unmon[:, 1:] - time_seq_unmon[:, :-1]

    #encoding of labels
    labels_mon = labels_mon - 1
    labels_mon_grouped = to_categorical(labels_mon // 10, num_classes=num_mon_sites // 10)
    labels_mon = to_categorical(labels_mon, num_classes=num_mon_sites)

    def build_histogram_256(dir_time, abs_time):
        tenth_packet_time = abs_time[10]
        abs_time = abs_time[10:]
        for i in range(len(abs_time)):
            abs_time[i] = abs_time[i] - tenth_packet_time
        dir_time = dir_time[10:]

        timing = np.subtract(np.geomspace(1, 50, num=257),1)
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

    lstm_data = []
    lstm_histogram_256 = []
    lstm_data_unmon = []
    lstm_histogram_256_unmon = []
    for i in range(len(time_seq_mon)):
        if(i%1000 == 0):
            print(i)
        
        lstm_trace = get_packet_timing(dir_seq_mon[i], abs_time_mon[i])
        lstm_hist_256 = build_histogram_256(dir_seq_mon[i], abs_time_mon[i])

        if(num_unmon_sites > 0):
            lstm_trace_unmon = get_packet_timing(dir_seq_unmon[i], abs_time_unmon[i])
            lstm_hist_256_unmon = build_histogram_256(dir_seq_unmon[i], abs_time_unmon[i])
            lstm_data_unmon.append(lstm_trace_unmon)
            lstm_histogram_256_unmon.append(lstm_hist_256_unmon)

        lstm_data.append(lstm_trace)
        lstm_histogram_256.append(lstm_hist_256)

    lstm_data = np.array(lstm_data)
    lstm_histogram_256 = np.array(lstm_histogram_256)

    lstm_data_unmon = np.array(lstm_data_unmon)
    lstm_histogram_256_unmon = np.array(lstm_histogram_256_unmon).reshape((num_unmon_sites, 256))
               
    grouped_data = []
    for lstm_data, lstm_histogram_256, labels, labels_grouped in zip(lstm_data, lstm_histogram_256, labels_mon, labels_mon_grouped):
        grouped_data.append((lstm_data, lstm_histogram_256, labels, labels_grouped))


    #shuffling data
    random.shuffle(grouped_data)

    #train/val/test split
    num_traces = num_mon_sites * num_mon_inst
    for i in range(5):
        #splitting into train/test/val for each of the 5 cross-validations
        block_size = int(num_traces / 5)
        test_set = grouped_data[(i%5)*block_size:((i)%5)*block_size + block_size]
        val_set = grouped_data[((i+1)%5)*block_size:((i+1)%5)*block_size + block_size]
        train_set = grouped_data[((i+2)%5)*block_size:((i+2)%5)*block_size + block_size] + grouped_data[((i+3)%5)*block_size:((i+3)%5)*block_size+block_size] + grouped_data[((i+4)%5)*block_size:((i+4)%5)*block_size + block_size]

        train_dir = np.array([x[0] for x in train_set])
        train_hist_256 = np.array([x[1] for x in train_set])
        train_labels = np.array([x[2] for x in train_set])
        train_labels_grouped = np.array([x[3] for x in train_set])

        val_dir = np.array([x[0] for x in val_set])
        val_hist_256 = np.array([x[1] for x in val_set])
        val_labels = np.array([x[2] for x in val_set])
        val_labels_grouped = np.array([x[3] for x in val_set])

        test_dir = np.array([x[0] for x in test_set])
        test_hist_256 = np.array([x[1] for x in test_set])
        test_labels = np.array([x[2] for x in test_set])
        test_labels_grouped = np.array([x[3] for x in test_set])

        with open(output_dir + 'lstm_train_dir_' + str(i) + '.npy','wb') as f:
            np.save(f, train_dir)

        with open(output_dir + 'lstm_val_dir_' + str(i) + '.npy', 'wb') as f:
            np.save(f, val_dir)

        with open(output_dir + 'lstm_test_dir_' + str(i) + '.npy', 'wb') as f:
            np.save(f, test_dir)

        with open(output_dir + 'lstm_train_hist_256_' + str(i) + '.npy','wb') as f:
            np.save(f, train_hist_256)

        with open(output_dir + 'lstm_val_hist_256_' + str(i) + '.npy', 'wb') as f:
            np.save(f, val_hist_256)

        with open(output_dir + 'lstm_test_hist_256_' + str(i) + '.npy', 'wb') as f:
            np.save(f, test_hist_256)
    
        with open(output_dir + 'lstm_train_labels_' + str(i) + '.npy', 'wb') as f:
            np.save(f, train_labels)

        with open(output_dir + 'lstm_val_labels_' + str(i) + '.npy', 'wb') as f:
            np.save(f, val_labels)

        with open(output_dir + 'lstm_test_labels_' + str(i) + '.npy', 'wb') as f:
            np.save(f, test_labels)
        
        with open(output_dir + 'lstm_train_labels_grouped_' + str(i) + '.npy', 'wb') as f:
            np.save(f, train_labels_grouped)

        with open(output_dir + 'lstm_val_labels_grouped_' + str(i) + '.npy', 'wb') as f:
            np.save(f, val_labels_grouped)

        with open(output_dir + 'lstm_test_labels_grouped_' + str(i) + '.npy', 'wb') as f:
            np.save(f, test_labels_grouped)

    with open(output_dir + 'lstm_timing_unmon.npy', 'wb') as f:
        np.save(f, lstm_data_unmon)
    
    with open(output_dir + 'lstm_histogram_unmon.npy', 'wb') as f:
        np.save(f, lstm_histogram_256_unmon)


if __name__ == '__main__':
    main()
