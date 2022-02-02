import sys
import os
import numpy as np
import pandas as pd


IO_READ = 1
IO_WRITE = 0


def get_req_info(raw_trace_path, group_num):
    raw_data = pd.read_csv(raw_trace_path, dtype='float32', sep=' ', header=None)
    raw_read_data = raw_data[raw_data[4] == IO_READ]
    max_group_num = int(len(raw_read_data) / 100)
    group_num = min(max_group_num, group_num)
    timeStamp = raw_read_data[0].values
    size = ((raw_read_data[3].values / 512 + 7) / 8).astype(np.int16)
    print(size)
    # size = int((int(size) / 512 + 7) / 8)
    req_list = []
    # print(raw_read_data.shape[0])
    for i in range(group_num):
        start_idx = raw_read_data.shape[0] - (group_num - i) * 100 - 1
        end_idx = start_idx + 100
        # print(end_idx)
        iops = (100.0/ (timeStamp[end_idx] - timeStamp[start_idx])*1.0)
        print(iops)
        io_sizes = np.asarray(size[start_idx: end_idx]).astype("float32")
        req_list.append(np.insert(io_sizes, 0, iops))

    req_array = np.asarray(req_list)
    # print(req_array)
    return req_array


def get_latency_info(replayed_trace_path, group_num, l_type='prev'):
    replayed_data = pd.read_csv(replayed_trace_path, dtype='float32', sep=',', header=None)
    replayed_read_data = replayed_data[replayed_data[2] == IO_READ]
    max_group_num = int(len(replayed_read_data)/100)
    group_num = min(max_group_num, group_num)
    ori_latency = replayed_read_data[1].values
    ###########################################
    start = replayed_read_data.shape[0] - (group_num) * 100 - 1
    print("start: ", start)

    ###########################################
    lat_list = []
    if l_type == 'prev':
        for i in range(group_num):
            end_idx = replayed_read_data.shape[0] - (group_num - i) * 100 - 1
            # print(end_idx)

            latency = ori_latency[:end_idx]

            p40 = np.percentile(latency, 40)
            p50 = np.percentile(latency, 50)
            p60 = np.percentile(latency, 60)
            p70 = np.percentile(latency, 70)
            p80 = np.percentile(latency, 80)
            p90 = np.percentile(latency, 90)
            p95 = np.percentile(latency, 95)
            p97 = np.percentile(latency, 97)
            lat_list.append([p40, p50, p60, p70, p80, p90, p95]/p95)
            print(p40, p50, p60, p70, p80, p90, p95)
        lat_array = np.asarray(lat_list)
    else:
        for i in range(group_num):
            end_idx = replayed_read_data.shape[0] - (group_num - i - 1) * 100 - 1
            # print(end_idx)
            latency = ori_latency[:end_idx]
            p40 = np.percentile(latency, 40)
            p50 = np.percentile(latency, 50)
            p60 = np.percentile(latency, 60)
            p70 = np.percentile(latency, 70)
            p80 = np.percentile(latency, 80)
            p90 = np.percentile(latency, 90)
            p95 = np.percentile(latency, 95)
            p97 = np.percentile(latency, 97)
            lat_list.append([p40, p50, p60, p70, p80, p90, p95] / p95)
            lat_array = np.asarray(lat_list)
            lat_array = lat_array.squeeze()
    # print(lat_array.shape)
    return lat_array


def feature_extractor(raw_trace_path, replayed_trace_path, group_num):
    req_array = get_req_info(raw_trace_path, group_num)
    lat_array_input = get_latency_info(replayed_trace_path, l_type='prev', group_num=group_num)
    output_data = get_latency_info(replayed_trace_path, l_type='output', group_num=group_num)
    input_data = []
    input_data.extend(np.hstack((req_array, lat_array_input)))
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    # print(input_data.shape)
    # print(output_data.shape)
    traceType = replayed_trace_path.split('/')[-1].split('.')[0]
    editOption = replayed_trace_path.split('/')[-2]
    editOptions = ['', 'out-rerated-10.0', 'out-rerated-100.0', 'out-resized-10.0', 'out-resized-100.0']
    if editOption not in editOptions:
        editOption = ''
    if not os.path.exists('dataset_7'):
        os.makedirs('dataset_7/input/')
        os.makedirs('dataset_7/output/')
    print("input size:", input_data.shape)
    print("output size: ", output_data.shape)
    pd.DataFrame(input_data).to_csv('dataset_7/input/data_'+traceType+'_'+editOption+'.csv', header=False, index=False)
    pd.DataFrame(output_data).to_csv('dataset_7/output/data_'+traceType+'_'+editOption + '.csv', header=False, index=False)


if len(sys.argv) < 2:
    print('illegal cmd format')
    exit(1)
print('raw_trace_path:', sys.argv[1])
print('replayed_trace_path:', sys.argv[2])

feature_extractor(sys.argv[1], sys.argv[2], group_num=600)

# req_array = get_req_info("../TRACETOREPLAY/msft-most-iops-10-mins/azure.trace")
# lat_array = get_latency_info('../TRACEPROFILE/nvme0n1/azure.trace', type='prev')
# input_array = np.hstack((req_array, lat_array))
# print(input_array.shape)


# IOPS: 4 digits
# size