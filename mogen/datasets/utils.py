from functools import cache
import numpy as np



@ cache
def p_def(x:int):
    if x < 0:
        raise ValueError(f'Value {x} cannot be negative.')
    elif x < 3:
        return 1
    elif x < 10:
        return -0.1 * x + 1.3
    else:
        return 0.3


def random_select_stickman(length:int, gap:int=4):
    # max_length = 196 
    candidate = []
    if length < (2 * gap + 1):
        return candidate
    mask = np.zeros(length, dtype=np.int32)
    # max_candidate = length // gap
    max_candidate = length // (2 * gap + 1)
    # weight = np.array([1,0.5,0.5]+[0.2 for i in range(1, max_candidate-1)])
    if max_candidate == 0:
        raise ValueError(f'Cannot select candidates with length {length} and gap {gap}.')
    else:
        weight = np.array([p_def(i) for i in range(max_candidate+1)])
        
    weight = weight / sum(weight)
    candidate_num = np.random.choice(range(max_candidate+1), p=weight)
    for i in range(candidate_num):
        zeros_indices = np.where(mask == 0)[0]
        if len(zeros_indices) == 0:
            raise ValueError(f'No available indices to select from mask {mask}.')
        selected_index = np.random.choice(zeros_indices)
        candidate.append(selected_index)
        left = max(0, selected_index - gap)
        right = min(length, selected_index + gap + 1)
        mask[left:right] = 1  # Mark the range as occupied
    candidate.sort()
    return candidate

if __name__ == "__main__":
    for lens in range(10, 196):
        for i in range(100):
            print(random_select_stickman(lens, 4))