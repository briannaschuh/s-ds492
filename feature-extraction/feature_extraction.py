import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import math
from Levenshtein import distance as levenshtein_distance

# calculate the minimum levenshtein distance between a password and a dataset of weak passwords
def min_levenshtein_distance(password, weak_passwords):
    min_distance = min(levenshtein_distance(password, weak_password) for weak_password in weak_passwords) #calculate the minumum levenshtein distance
    return min_distance

# count the different type of characters
def count_chars(password):
    uppercase = sum(1 for c in password if c.isupper()) # uppercase
    lowercase = sum(1 for c in password if c.islower()) # lowercase
    digits = sum(1 for c in password if c.isdigit()) # digits
    special = sum(1 for c in password if not c.isalnum()) # special 
    return uppercase, lowercase, digits, special

# calculate the password entropy
def password_entropy(password):
    if len(password) == 0: # if the password has length 0, return 0
        return 0 
    char_freq = Counter(password) # count the number of elements in a password
    entropy = -sum(freq / len(password) * math.log2(freq / len(password)) for freq in char_freq.values()) # calculate the entropy
    return entropy

# calculate the n-gram frequency 
def ngram_frequency(password, n):
    ngrams = [password[i:i + n] for i in range(len(password) - n + 1)] # store the n-grams
    ngram_count = len(ngrams) # count the n-grams
    unique_ngram_count = len(set(ngrams)) # get the unique count
    return unique_ngram_count / ngram_count if ngram_count > 0 else 0

# extract the features from the passwords
def extract_features(data_filepath, batch_size=10000):
    data = pd.read_csv(data_filepath, error_bad_lines=False) # read in the csv file
    data = data.dropna() # remove any empty or NaN password entries
    common_passwords = pd.read_csv("data/common_passwords.csv") # store the dataframe into a variable
    common_passwords_list = common_passwords["password"].tolist() # convert it to a list
    extracted_features_csv = data_filepath[:-4] + "_featureExtracted.csv" # generate the name of the output csv file
    header = ["password", "length", "uppercase", "lowercase", "digits", "special", "entropy", "bigram_freq", "trigram_freq", "fourgram_freq", "levenshtein_distance"] # create the headers of the csv file
    with open(extracted_features_csv, "w") as f_out: # open the file containing the features
        f_out.write(",".join(header) + "\n") # write the headers

    scaler = MinMaxScaler() # initialize the MinMaxScaler
    max_weak_password_length = max(len(p) for p in common_passwords_list) # find the length of the longest password in the common passwords list
    num_batches = len(data) // batch_size + 1 # process the dataset in batches
    for batch_idx in range(num_batches): # iterate through the number of batches
        start_idx = batch_idx * batch_size # calculate the start index of the batch 
        end_idx = min((batch_idx + 1) * batch_size, len(data)) #calculate the end index of the batch
        batch_data = data.iloc[start_idx:end_idx] #subset the dataframe
        features_list = [] # initalize the list of features
        for index, row in batch_data.iterrows(): # iterate through each index and row
            password = row["password"] # store the password in a temporary variable
            length = len(password) # store the legth of the password
            uppercase, lowercase, digits, special = count_chars(password) # get the number of uppercase characters, lowercase characters, digits, and special characters of the password
            entropy = password_entropy(password) # calculate the password entropy
            bigram_freq = ngram_frequency(password, 2) # calculate bi-gram
            trigram_freq = ngram_frequency(password, 3) # calculate tri-gram
            fourgram_freq = ngram_frequency(password, 4) # calculate four-gram
            levenshtein_distance = min_levenshtein_distance(password, common_passwords_list) # calculate the levenshtein distance
            levenshtein_distance_normalized = levenshtein_distance / max(len(password), max_weak_password_length) # normalize the levenshtein distance
            features = [length, uppercase, lowercase, digits, special, entropy, bigram_freq, trigram_freq, fourgram_freq, levenshtein_distance_normalized] # store the features in a list
            features_list.append(features) # append the features to the list
        features_array = np.array(features_list) # convert the list into an array
        features_to_scale = features_array[:, :-1] # scale the features but exclude the last feature (levenshtein_distance_normalized) since it was already scaled
        scaler.partial_fit(features_to_scale) # fits the scaler based on minimum and maximum values based on the most recent batch
        features_normalized = scaler.transform(features_to_scale) # apply the scaler to features
        for i, row in enumerate(batch_data.iterrows()): # add the features to the output file
            password = row[1]["password"] # store the password
            with open(extracted_features_csv, "a") as f_out: # open the password
                f_out.write(f"{password},{','.join(map(str, features_normalized[i]))},{features_array[i, -1]}\n") # write to the file

