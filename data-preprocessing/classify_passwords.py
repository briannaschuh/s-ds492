import csv

import re
import csv

import zxcvbn

def classification(password):
    result = zxcvbn.zxcvbn(password)
    score = result['score']

    
    if score == 0 or score == 1:
        return "Weak"
    elif score == 2:
        return "Medium"
    elif score == 3:
        return "Strong"
    elif score == 4:
        return "Very Strong"


def classify_passwords(data_filepath):
    with open(data_filepath, newline='') as input_file:
        reader = csv.reader(input_file) # create a csv reader
        output = data_filepath[:-4] + "_classifed.csv" # name of the output file
        with open(output, 'w', newline='') as output_file: # create the output file
            writer = csv.writer(output_file)
            writer.writerow(['password', 'strength']) # write the name of the headers
        for row in reader: # iterate through each row
            password = row[0] # extract the password
            strength = classification(password) # apply the classification function
            with open(output, 'a', newline='') as output_file: # open the output
                writer = csv.writer(output_file)  # open writer
                writer.writerow([password, strength]) # write the password and the strength
    print("The passwords have been categorized. ")

