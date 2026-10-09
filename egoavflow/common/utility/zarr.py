import concurrent.futures

import numpy as np
import zarr


def parallel_reading(
    group: zarr.hierarchy.Group,
    array_name: str,
    max_workers=48,
):
    zarr_arr = group[array_name]
    zarr_arr_shape = zarr_arr.shape

    # create empty numpy array
    np_array = np.empty(zarr_arr_shape, dtype=zarr_arr.dtype)

    def copy(zarr_arr, zarr_idx, np_array, np_idx):
        try:
            np_array[np_idx] = zarr_arr[zarr_idx]
            # make sure we can successfully read
            _ = np_array[zarr_idx]
            return True
        except Exception as e:
            print(e)
            return False

    n = zarr_arr_shape[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for i in range(n):
            futures.add(
                executor.submit(
                    copy,
                    zarr_arr,
                    i,
                    np_array,
                    i,
                )
            )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    return np_array
