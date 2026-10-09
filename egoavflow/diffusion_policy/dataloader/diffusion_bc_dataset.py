
import numpy as np


normalize_threshold = 1e-8


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length): # something is clipped due to horizon
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0: # pad past
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length: # pad future
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0), 'mean': np.mean(data, axis=0), 'std': np.std(data, axis=0)}
    return stats


def normalize_data(data, stats, type):
    if type == 'min_max':
        # nomalize to [0,1]
        ndata = data.copy()
        for i in range(ndata.shape[-1]):
            if stats["max"][i] - stats["min"][i] > normalize_threshold:
                ndata[..., i] = (data[..., i] - stats["min"][i]) / (
                    stats["max"][i] - stats["min"][i]
                )
                # normalize to [-1, 1]
                ndata[..., i] = ndata[..., i] * 2 - 1
    elif type == 'mean_std':
        ndata = data.copy()
        for i in range(ndata.shape[-1]):
            ndata[..., i] = (data[..., i] - stats["mean"][i]) / (stats["std"][i] + 1e-10)
    else:
        raise NotImplementedError(f"Invalid norm type: {type}")
    return ndata


def unnormalize_data(ndata, stats, type):
    if type == 'min_max':
        data = ndata.copy()
        for i in range(ndata.shape[-1]):
            if stats["max"][i] - stats["min"][i] > normalize_threshold:
                ndata[..., i] = (ndata[..., i] + 1) / 2
                data[..., i] = (
                    ndata[..., i] * (stats["max"][i] - stats["min"][i]) + stats["min"][i]
                )
    elif type == 'mean_std':
        data = ndata.copy()
        for i in range(ndata.shape[-1]):
            data[..., i] = ndata[..., i] * (stats["std"][i]) + stats["mean"][i]
    else:
        raise NotImplementedError(f"Invalid norm type: {type}")
    return data

