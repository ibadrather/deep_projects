"""
    Author: Ibad Rather, ibad.rather.ir@gmail.com (July 2022)

    ## Dataset Details:

    I will generate different vibrations randomly. Each vibration will have
    three characteristics/parameters.

        A  -> Amplitude:    The Amplitude values will be in range: (10e-3, 10e2) meters.
        tc -> Time Constant:    The Time Constant values will be in range: (1, 10e2) seconds 
                                10e3 means almost undamped wave -> atleast for the given time 
                                period of t = 25 seconds.

        omega -> Angular Frequency: The Angular Frequency values will be in three different ranges
                    1. low: (1, 7) rad/s 
                    2. medium: (15, 22) rad/s 
                    3. high: (30, 37) rad/s 
    
    The oscillation will happen over a time period of t = 8 seconds.
    The Oscillation will have a data frequency of 100 Hz.

    ## Purpose:
    First I will generate different types of oscillating signals()

    The goal of generating this dataset is to classify a given oscilating signal or wave into one of
    three angular frequency categories: low, medium and high.

    I believe this will be a challenging task as there are many factors involved here and a combination
    of these factors will generate a lot of different types of oscillating waves.

    Let's see and discover which neural networks are able to solve this classification problem.

    The mathematical equation of the oscillation will be

    # Half of them will be sin waves and half of them cos waves
    y = A * e^(-timestamp/tc) * sin(omega * timestamp)
        and
    ! Not Implemented: y = A * e^(-timestamp/tc) * cos(omega * timestamp)

    
"""

import numpy as np
import numpy.random as random
import pandas as pd
from tqdm import tqdm
import os

try:
    os.system("clear")
except:
    pass

# Num of samples we want to generate
num_samples = 5000  # int(15e2)

assert num_samples % 10 == 0, "Num of samples should be divisible by 10"

# Timestamp array
t = 8  # seconds
data_frequency = 100  # Hz
timestamps = np.linspace(0, t, t * data_frequency)

# Generating oscillation parameters
amplitudes = random.uniform(10e-3, 10e2, num_samples)
time_constants = random.uniform(1, 10e2, num_samples)

# Now we generate the angular frequencies in the low, medium and high range and their labels
# Low Frequeny data
low_frequencies = random.uniform(0.5, 2, num_samples // 3)
low_freq_labels = np.full_like(low_frequencies, 1, dtype=int)

# Medium frequency data
medium_frequencies = random.uniform(8, 14, num_samples // 3)
medium_freq_labels = np.full_like(medium_frequencies, 2, dtype=int)

# High frequency data
high_frequencies = random.uniform(
    25, 30, num_samples // 3 + 1
)  # adding 1 because multiples of 10 not divisible by 3
high_freq_labels = np.full_like(high_frequencies, 3, dtype=int)

# stacking these three frequency and label arrays together
angular_frequencies = np.hstack((low_frequencies, medium_frequencies, high_frequencies))
labels = np.hstack((low_freq_labels, medium_freq_labels, high_freq_labels))


# Generating Oscillations
oscillations = []
class_labels = []
for i, (amplitude, time_constant, angular_frequency) in tqdm(
    enumerate(zip(amplitudes, time_constants, angular_frequencies))
):
    oscillation = (
        amplitude
        * np.exp(-timestamps / time_constant)
        * np.sin(angular_frequency * timestamps)
    )
    label = labels[i]

    oscillations.append(oscillation)
    class_labels.append(label)


# Converting into Numpy Arrays
oscillations = np.array(oscillations)
class_labels = np.array(class_labels)


# Let's visualise a single oscillation in each class
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.tight_layout()
# fig.suptitle('Oscillations of Different Frequencies')
ax1.plot(oscillations[0])
ax1.set_title("Low Frequency")

ax2.plot(oscillations[oscillations.shape[0] // 3 + 10])
ax2.set_title("Medium Frequency")

ax3.plot(oscillations[-1])
ax3.set_title("High Frequency")

plt.savefig("3_classes_oscillations.png", dpi=300)
# plt.show()
plt.close("all")

# Creating a DataFrame of this Data
features = pd.DataFrame(oscillations)
labels = pd.DataFrame(labels)
labels.columns = ["class_label"]

dataset = pd.concat([features, labels], axis=1)
print("Dataset Shape: ", dataset.shape)

# Randomly shuffling this dataset before saving
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Giving lalel names
dataset["class_label"] = dataset["class_label"].replace([1], "low")
dataset["class_label"] = dataset["class_label"].replace([2], "medium")
dataset["class_label"] = dataset["class_label"].replace([3], "high")

# print(dataset.head())

# Train, Val and Test Split
def train_validate_test_split(df, train_percent=0.8, validate_percent=0.15, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


train, validate, test = train_validate_test_split(
    dataset, train_percent=0.8, validate_percent=0.15, seed=42
)

train, validate, test = train.dropna(), validate.dropna(), test.dropna()

print("Len Train Dataset: ", len(train))
print("Len Validate Dataset: ", len(validate))
print("Len Test Dataset: ", len(test))

train.to_csv("train_oscillation.csv", index=False)
validate.to_csv("val_oscillation.csv", index=False)
test.to_csv("test_oscillation.csv", index=False)

print("Process Complete.")
print("Datasets Generated and Saved as .csv's")

## Sanity checking the data

print(train.isnull().values.any())
print(validate.isnull().values.any())
print(test.isnull().values.any())


print(train.isnull().sum().sum())
print(validate.isnull().sum().sum())
print(test.isnull().sum().sum())
