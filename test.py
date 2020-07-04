import numpy as np

scale_bases = [2, 16]
scale_steps = 7
num_frames = 128


def construct_proposals():
    all_props = []
    for scale_idx, scale_base in enumerate(scale_bases):
        current_starts = []
        current_ends = []
        for rep in range(2, scale_steps + 2):
            starts = np.arange(0, num_frames - scale_base * rep + 1, scale_base)
            ends = starts + scale_base * rep
            current_starts.append(starts)
            current_ends.append(ends)
        all_props.append(np.stack((np.concatenate(current_starts, axis=0),
                                   np.concatenate(current_ends, axis=0)), axis=1))
    return np.concatenate(all_props, axis=0)

if __name__ == "__main__":
    props = construct_proposals()
    for start, end in props:
        print((start * 2, end * 2))
    # print(props)
    print(len(props))