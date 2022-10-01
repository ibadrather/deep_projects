import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
dataset = pd.read_csv('BankNote_Authentication.csv')

# Split into train and test sets
train, test = train_test_split(dataset, test_size=0.05, random_state=1, shuffle=True)

print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

# Save the train and test sets
train.to_csv('BankNote_Authentication_train.csv', index=False)
test.to_csv('BankNote_Authentication_test.csv', index=False)
