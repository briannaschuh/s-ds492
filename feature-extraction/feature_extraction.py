import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import math
import string
from decimal import Decimal
from Levenshtein import distance as levenshtein_distance

# calculate the minimum levenshtein distance between a password and a dataset of weak passwords
def min_levenshtein_distance(password, weak_passwords):
    return min(levenshtein_distance(password, weak_password) for weak_password in weak_passwords) #calculate the minumum levenshtein distance

# calculate character repetition
def character_repetition(password):
    repetitions = 0 # keep track of repetitions
    weight_sum = 0 # used to measure how severe a repetition was
    current_repetition = 1 # initalize
    for i in range(1, len(password)): # iterate through each character
        if password[i] == password[i - 1]: # if the passwords are the same, increase the repetition
            current_repetition += 1
        else: # adjust the weight if there is no repetition
            weight_sum += current_repetition * (current_repetition - 1)
            repetitions += current_repetition - 1
            current_repetition = 1
    weight_sum += current_repetition * (current_repetition - 1)
    repetitions += current_repetition - 1
    return repetitions, weight_sum

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
    return -sum(freq / len(password) * math.log2(freq / len(password)) for freq in char_freq.values()) # calculate the entropy

# calculate the n-gram frequency 
def ngram_frequency(password, n):
    ngrams = [password[i:i + n] for i in range(len(password) - n + 1)] # store the n-grams
    ngram_count = len(ngrams) # count the n-grams
    unique_ngram_count = len(set(ngrams)) # get the unique count
    return unique_ngram_count / ngram_count if ngram_count > 0 else 0

# calculate how long it would take for an attacker to crack the password by brute force
def cracking_time(password, attempts_per_second = 10000000000):
    PASSWORD_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-=_+[]|;':\",./<>?"
    key_space = len(PASSWORD_CHARSET) ** len(password)  # calculate the total key space
    total_combinations = Decimal(key_space)  # convert the key_space to a Decimal
    attempts_per_second = Decimal(attempts_per_second)  # convert the attempts_per_second to a Decimal
    time_to_crack_seconds = total_combinations / attempts_per_second  # calculate the time to crack in seconds
    time_to_crack = convert_time_units(time_to_crack_seconds)  # format the time according to the specified thresholds
    return time_to_crack

# convert seconds to the appropriate time unit
def convert_time_units(time_in_seconds):
    seconds_per_minute = Decimal(60) # seconds per minute
    minutes_per_hour = Decimal(60) # minutes per hour
    hours_per_day = Decimal(24) # hours per day
    days_per_week = Decimal(7) # days per week
    days_per_month = Decimal(30.44)  # average number of days in a month
    days_per_year = Decimal(365.25)  # average number of days in a year, considering leap years
    if time_in_seconds >= 5 * days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute:  # if seconds is greater than the number of seconds in 5 years
        return "5+ years"
    if time_in_seconds >= days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute: # if seconds is greater than the number of seconds in one year
        time_in_years = time_in_seconds / (days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute)
        return f"{time_in_years:.2f} years"
    if time_in_seconds >= days_per_week * hours_per_day * minutes_per_hour * seconds_per_minute: # if seconds is greater than the number of seconds in one week
        time_in_weeks = time_in_seconds / (days_per_week * hours_per_day * minutes_per_hour * seconds_per_minute)
        return f"{time_in_weeks:.2f} weeks"
    if time_in_seconds >= hours_per_day * minutes_per_hour * seconds_per_minute: # if seconds is greater than the number of seconds in one day
        time_in_days = time_in_seconds / (hours_per_day * minutes_per_hour * seconds_per_minute)
        return f"{time_in_days:.2f} days"
    if time_in_seconds >= minutes_per_hour * seconds_per_minute: # if seconds is greater than the number of seconds in one hour
        time_in_hours = time_in_seconds / (minutes_per_hour * seconds_per_minute)
        return f"{time_in_hours:.2f} hours"
    if time_in_seconds >= seconds_per_minute: # if seconds is greater than the number of seconds in one miute
        time_in_minutes = time_in_seconds / seconds_per_minute
        return f"{time_in_minutes:.2f} minutes"
    return f"{time_in_seconds:.2f} seconds" # if neither of these conditions hold then return the seconds as is

# extract the features from the passwords
def extract_features(data_filepath, batch_size=10000):
    data = pd.read_csv(data_filepath, error_bad_lines=False) # read in the csv file
    data = data.dropna() # remove any empty or NaN password entries
    #column_names = ['password', 'strength']  # Define the column names
    #data = pd.read_csv(data_filepath, error_bad_lines=False, names=column_names)  # Add 'names' parameter
    #data = data.dropna()  # remove any empty or NaN password entries
    common_passwords = pd.read_csv("data/common_passwords.csv") # store the dataframe into a variable
    common_passwords_list = common_passwords["password"].tolist() # convert it to a list
    extracted_features_csv = data_filepath[:-4] + "_featureExtracted.csv" # generate the name of the output csv file
    header = ["password", "strength", "length", "uppercase", "lowercase", "digits", "special", "entropy", "bigram_freq", "trigram_freq", "fourgram_freq", "levenshtein_distance", "char_repetition_weight_sum", "cracking_time"] # create the headers of the csv file
    with open(extracted_features_csv, "w") as f_out: # open the file containing the features
        f_out.write(",".join(header) + "\n") # write the headers
    scaler = MinMaxScaler() # initialize the MinMaxScaler
    max_common_password_length = len(max(common_passwords_list)) # find the length of the longest password in the common passwords list
    num_batches = len(data) // batch_size + 1 # process the dataset in batches
    for batch_idx in range(num_batches): # iterate through the number of batches
        start_idx = batch_idx * batch_size # calculate the start index of the batch 
        end_idx = min((batch_idx + 1) * batch_size, len(data)) #calculate the end index of the batch
        batch_data = data.iloc[start_idx:end_idx] #subset the dataframe
        features_list = [] # initalize the list of features
        for index, row in batch_data.iterrows(): # iterate through each index and row
            password = row["password"] # store the password in a temporary variable
            length = len(password) # store the length of the password
            uppercase, lowercase, digits, special = count_chars(password) # get the number of uppercase characters, lowercase characters, digits, and special characters of the password
            entropy = password_entropy(password) # calculate the password entropy
            bigram_freq = ngram_frequency(password, 2) # calculate bi-gram
            trigram_freq = ngram_frequency(password, 3) # calculate tri-gram
            fourgram_freq = ngram_frequency(password, 4) # calculate four-gram
            levenshtein_distance = min_levenshtein_distance(password, common_passwords_list) # calculate the levenshtein distance
            levenshtein_distance_normalized = levenshtein_distance / max(len(password), max_common_password_length) # normalize the levenshtein distance
            repetition, char_repetition_weight_sum = character_repetition(password)  # calculate character repetition weight sum
            cracking_time_estimate = cracking_time(password, 10_000_000_000)  # calculate estimated cracking time
            features = [length, uppercase, lowercase, digits, special, entropy, bigram_freq, trigram_freq, fourgram_freq, levenshtein_distance_normalized, char_repetition_weight_sum, cracking_time_estimate] # store the features in a list
            features_list.append(features) # append the features to the list
        features_array = np.array(features_list) # convert the list into an array
        features_to_scale = features_array[:, :-2] # scale the features but exclude the last feature (levenshtein_distance_normalized) since it was already scaled
        scaler.partial_fit(features_to_scale) # fits the scaler based on minimum and maximum values based on the most recent batch
        features_normalized = scaler.transform(features_to_scale) # apply the scaler to features
        for index, row in enumerate(batch_data.iterrows()): # add the features to the output file
            password = row[1]["password"] # store the password
            strength = row[1]["strength"] # store the strength
            with open(extracted_features_csv, "a") as f_out: # open the password
                f_out.write(f"{password},{strength},{','.join(map(str, features_normalized[index]))},{features_array[index, -2]},{features_array[index, -1]}\n") # write to the file

    print("We have extracted all of the features")

