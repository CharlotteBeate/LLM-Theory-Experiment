#################################################################################
# This script is showing the strategic agents in a simplified context (2 token types, 2 autoregressive steps)
# how they can manipulate the probabilities of the paths surviving the first autoregressive step depending on the strategic actors modified proabilities
# ################################################################################


# ################################################################################
# Import packages
import numpy as np
from icecream import ic

# ################################################################################
# Define which experiments are run
__three__ = False
__four__ = False
__five__ = False
__six__ = True

#################################################################################
# Setup
#################################################################################
# Define the temperature
T = 1
# Define the softmax transformation
def transformation(P, T):
    P_transformed = np.power(P, 1/T)
    return P_transformed / np.sum(P_transformed)


def gen_prob(n):
    random_numbers = np.random.random(n)
    starting_probabilities1 = random_numbers / np.sum(random_numbers)
    starting_probabilities1 = np.round(starting_probabilities1, 3)
    starting_probabilities1[-1] += 1 - np.sum(starting_probabilities1)
    return starting_probabilities1


def modify_vector(vector, n, m):
    # Make sure vector is a numpy array
    vector = np.array(vector)

    # Check if n and m are valid indices
    if n < 0 or n >= len(vector) or m < 0 or m >= len(vector):
        raise IndexError("Indices are out of range")

    # Add the n'th entry to the m'th entry
    vector[m] += vector[n]

    # Set the n'th entry to zero
    vector[n] = 0

    return vector

# Function: It takes a list as the input vector and 0-indexed positions as the n-th and m-th positions to set to zero. Then, it normalizes the remaining elements so that their sum is 1.
def zero_and_normalize(vec, n, m):
    vec1 = vec
    # Check if n and m are valid indices for the given vector
    if not (0 <= n < len(vec1) and 0 <= m < len(vec1)):
        raise ValueError("Indices n and m are out of bounds for the given vector.")

    # Set nth and mth positions to 0
    vec1[n] = vec1[m] = 0

    # Calculate the sum of the vector
    total = sum(vec1)

    # Check if the total is zero (i.e., all elements were zero or only the elements at n-th and m-th position were non-zero)
    if total == 0:
        print("The sum of the remaining elements after setting the n-th and m-th elements to zero is zero.")
        print("The function can't normalize the vector to sum up to 1.")
        return vec1

    # Scale the rest of the vector to sum up to 1
    new = vec1/total
    return new

def sum_elements(valid_p):
    # Check if valid_p has at least 4 elements
    if len(valid_p) < 4:
        raise ValueError("The list should have at least four elements.")

    # Add up the first and third element
    sum_1_3 = valid_p[0] + valid_p[2]

    # Add up the second and fourth element
    sum_2_4 = valid_p[1] + valid_p[3]

    return sum_1_3, sum_2_4

def duplicate(p_trans):
    # Check if p_trans has exactly 2 entries
    if len(p_trans) != 2:
        raise ValueError("The list should have exactly two elements.")

    # Duplicate the list
    new_p_trans = p_trans * 2

    return new_p_trans

def equalize_entries(vector, n, m):
    # Calculate the average of the nth and mth entries
    avg_value = (vector[n] + vector[m]) / 2
    
    # Set the nth and mth entries to the average value
    vector1 = vector.copy()
    vector1[n] = avg_value
    vector1[m] = avg_value
    
    return vector1

def resembling(vector):
    # Check if the vector has exactly 4 entries
    if len(vector) != 4:
        raise ValueError("Input vector should have exactly 4 entries.")
    
    # Check whether entry 4 (index 3) or entry 3 (index 2) is bigger
    if vector[3] > vector[2]:
        vector = modify_vector(vector, 0, 1)
    elif vector[2] > vector[3]:
        vector = modify_vector(vector, 1, 0)
    
    return vector

def diverging(vector):
    # Check if the vector has exactly 4 entries
    if len(vector) != 4:
        raise ValueError("Input vector should have exactly 4 entries.")
    
    # Check whether entry 4 (index 3) or entry 3 (index 2) is bigger
    if vector[3] < vector[2]:
        vector = modify_vector(vector, 0, 1)
        ic("increasing")
    elif vector[2] < vector[3]:
        vector = modify_vector(vector, 1, 0)
    
    return vector

def same_proportion(vector):
    vector_copy = vector.copy()
    # Check if the vector has exactly 4 entries
    if len(vector_copy) != 4:
        raise ValueError("Input vector should have exactly 4 entries.")
    
    # calculate the proportion of entry 4 to entry 3
    proportion_2 = vector_copy[2] / (vector_copy[2] + vector_copy[3])
    # Move mass from entry 1 to entry such that the proportion of 2 to 1 is the same as 4 to 3. 
    # Sum of vector[0] + vector[1] stays the same
    sum01 = vector_copy[0] + vector_copy[1]
    vector_copy[0] = proportion_2 * sum01
    vector_copy[1] = sum01 - vector_copy[0]
    #assert that vector[1] is close to (1 - proportion_2) * sum01
    assert np.isclose(vector_copy[1], (1 - proportion_2) * sum01)
    return vector_copy

def maj_min(vector):
    vector_copy = vector.copy()
    # Check if the vector has exactly 4 entries
    if len(vector_copy) != 4:
        raise ValueError("Input vector should have exactly 4 entries.")
    sum01 = vector_copy[0] + vector_copy[1]
    if sum01 > 0.5:
        vector_copy = resembling(vector_copy)
    elif sum01 < 0.5:
        vector_copy = diverging(vector_copy)
    return vector_copy

# ################################################################################
# Define the main function
# ################################################################################

def calculate_probabilities(p, p_m, paths, T):
    table_m_rem = np.zeros((4,2))
    table_m_forget = np.zeros((4,2))
    table = np.zeros((4,2))
    assert len(p) == 4
    assert len(p_m) == 4
    assert np.sum(p) == 1
    assert np.sum(p_m) == 1
    assert T > 0
    # Calculate probabilities for i-th position
    # add up the first and third element of valid_p
    # add up the second and fourth element of valid_p
    p_zero, p_one = sum_elements(p)
    p_zero_m, p_one_m = sum_elements(p_m)


    # Calculate the sampling probabiliity
    p_trans = transformation([p_zero, p_one], T)
    p_trans_m = transformation([p_zero_m, p_one_m], T)
    # ic(p_trans)
    assert len(p_trans) == 2
    # concatenat p_trans to itself
    p_trans =  np.concatenate((p_trans, p_trans))
    p_trans_m = np.concatenate((p_trans_m, p_trans_m))
    assert len(p_trans) == 4
    assert len(p_trans_m) == 4
    # ic(p_trans)
    # ic(p_trans_m)
    # use values of [p_trans for first column of table]
    # ic(table[:,0])
    table[:,0] = p_trans
    table_m_rem[:,0] = p_trans_m
    table_m_forget[:,0] = p_trans_m


    #################################################################################
    # Calculate the probabilities of being selected in the secon step conditional on being selected in the first steo
    #################################################################################
    p_copy = p.copy()
    p_copy_m = p_m.copy()
    # Calculate the conditional probabilities conditional on observing a 0
    update_0 = zero_and_normalize(p_copy, 1, 3)
    update_0_m_rem = zero_and_normalize(p_copy_m, 1, 3)
    # Calculate the conditional probabilities conditional on observing a 1

    update_1 = zero_and_normalize(p, 0, 2)
    update_1_m_rem = zero_and_normalize(p_m, 0, 2)

    # allow for various sampling rules: 
    update_0_t = transformation(update_0, T)
    update_1_t = transformation(update_1, T)    
    update_0_m_rem_t = transformation(update_0_m_rem, T)
    update_1_m_rem_t = transformation(update_1_m_rem, T)    

    # ic(update_0_t)
    # ic(update_1_t)
    # combining
    assert len(update_1_m_rem) == 4
    assert len(update_0_m_rem) == 4
    assert len(update_1) == 4
    assert len(update_0) == 4

    # update_rem = 
    update = np.sum([update_0_t, update_1_t], axis=0)
    update_rem = np.sum([update_0_m_rem_t, update_1_m_rem_t], axis=0)

    assert len(update) == 4
    assert len(update_rem) == 4

    # assert close to 2 the values of all vectors
    assert np.isclose(sum(update), 2, atol=1e-6), "The sum of the update is not close to 2"
    assert np.isclose(sum(update_rem), 2, atol=1e-6), "The sum of update_rem is not close to 2"


    # Calculate the overall probabilities of each path: 
    table[:,1] = update
    table_m_rem[:,1] = update_rem
    table_m_forget[:,1] = update
    # ic(table)
    # ic(table_m_rem)
    # ic(table_m_forget)

    #################################################################################
    # Preparing the results
    #################################################################################   

    p_path = np.prod(table, axis=1)
    p_path_m_rem = np.prod(table_m_rem, axis=1)
    p_path_m_forget = np.prod(table_m_forget, axis=1)

    assert np.isclose(sum(p_path_m_forget), 1, atol=1e-6)
    assert np.isclose(sum(p_path_m_rem), 1, atol=1e-6)
    assert np.isclose(sum(p_path), 1, atol=1e-6)

    group = np.array([p_path[0] + p_path[1], p_path[2] + p_path[3]])
    group_m_rem = np.array([p_path_m_rem[0] + p_path_m_rem[1], p_path_m_rem[2] + p_path_m_rem[3]])
    group_m_forget = np.array([p_path_m_forget[0] + p_path_m_forget[1], p_path_m_forget[2] + p_path_m_forget[3]])

    assert len(group) == 2
    # ic(group)
    # ic(group_m_rem)
    # ic(group_m_forget)

    assert np.isclose(sum(group), 1, atol=1e-6)

    return group, group_m_rem, group_m_forget

# ################################################################################
# Preparation of the experiments - specificaiton of the variables
# ################################################################################

# speciy the path
paths = [[0, 1],
        [ 1, 1], 
        [0, 0], 
        [ 1, 0]]
N = 100 
x = len(paths)
assert x == 4

#################################################################################
# When T is small putting all eggs in one basket and model remembering
#################################################################################
T = 0.2
abc = 0 # counting the number of times the first group is smaller than the second group
numbers = np.zeros(N)
if __three__:
    for i in range(N):
        start_p = gen_prob(x)
        if np.sum(start_p) == 1: 
            mod_prob = modify_vector(start_p, 0,1)
            group, group_m_rem, group_m_forget = calculate_probabilities(start_p, mod_prob, paths, T)
            if group[0] <= group_m_rem[0]:
                abc += 1
            elif group[0] > group_m_rem[0]:  
                print("First group probabilities become smaller!")
                ic(start_p)
                ic(mod_prob)
                ic(group)
                ic(group_m_rem)
            numbers[i] = group_m_rem[0] - group[0]
    print(abc)
    mean = np.mean(numbers)
    ic(mean) # for T = 0.2: mean: 0.23861962391946345
#################################################################################
# When T is big  and model remembering, distribute probability
#################################################################################
abc = 0
T = 1.3
numbers = np.zeros(N)
if __four__:
    for i in range(N):
        start_p = gen_prob(x)
        if np.sum(start_p) == 1: 
            mod_prob = equalize_entries(start_p, 0,1)
            # mod_prob = same_proportion(start_p)
            # mod_prob = resembling(start_p)
            group, group_m_rem, group_m_forget = calculate_probabilities(start_p, mod_prob, paths, T)
            if group[0] <= group_m_rem[0]:
                abc += 1
            elif group[0] > group_m_rem[0]:  
                print("First group probabilities become smaller!")
                ic(start_p)
                ic(mod_prob)
                ic(group)
                ic(group_m_rem)
            numbers[i] = group_m_rem[0] - group[0]
    print(abc)
    mean_distribute = np.mean(numbers)
    ic(mean_distribute) # 0.007549057883773478
#################################################################################
# When T is big and model forgetting, distribute probability
#################################################################################

abc = 0
fail = 0
T = 0.2
N = 600
numbers = np.zeros(N)
if __five__:
    print("experiment 5" )
    for i in range(N):
        start_p = gen_prob(x)
        if np.sum(start_p) == 1: 
            # mod_prob = equalize_entries(start_p, 0,1) # 24%
            # mod_prob = diverging(start_p)  # 70-85%
            mod_prob = maj_min(start_p)  # 70-85%
            # mod_prob = resembling(start_p)  # 25%
            # mod_prob = same_proportion(start_p) # 0% 

            group, group_m_rem, group_m_forget = calculate_probabilities(start_p, mod_prob, paths, T)
            if group[0] <= group_m_forget[0]:
                abc += 1
            elif group[0] > group_m_forget[0]:  
                print("First group probabilities become smaller!")
                fail += 1
                ic(start_p)
                ic(mod_prob)
                ic(group)
                ic(group_m_rem)
            numbers[i] = group_m_rem[0] - group[0]
    print(abc, fail)
    mean = np.mean(numbers)
    ic(mean) # 0.2876046634943488  for T = 0.2


abc = 0
fail = 0
T = 2.5
N = 100
numbers = np.zeros(N)
if __six__:
    print("experiment 6")
    for i in range(N):
        start_p = gen_prob(x)
        if np.sum(start_p) == 1: 
            # mod_prob = equalize_entries(start_p, 0,1) # 24%
            mod_prob = maj_min(start_p)  # 70-85%
            # mod_prob = diverging(start_p)  # 75%
            # mod_prob = resembling(start_p)  # 25%
            # mod_prob = same_proportion(start_p) # 1%  - 99% 
            group, group_m_rem, group_m_forget = calculate_probabilities(start_p, mod_prob, paths, T)
            if group[0] <= group_m_forget[0]:
                abc += 1
            elif group[0] > group_m_forget[0]:  
                print("First group probabilities become smaller!")
                fail += 1
                ic(start_p)
                ic(mod_prob)
                ic(group)
                ic(group_m_rem)
            numbers[i] = group_m_rem[0] - group[0]
    print(abc, fail)
    mean = np.mean(numbers)
    ic(numbers)
    ic(mean) #0.00778457745233036 for T = 2.5 (probably not anything significa,t sometimes negative)
            # 0.010259345666539823 for equalize_entries

# TODOs: 
# TODO: allow for some simple rule: 1) if smaller than 50%, then do something else
        #backfired

# ################################################################################
# Writing unit tests
# ################################################################################

# TODO: write unit tests for the functions
