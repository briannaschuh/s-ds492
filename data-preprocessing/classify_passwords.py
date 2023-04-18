import re
import csv

def classification(password):
    score = 0
    # rule 1: length
    if len(password) < 8:
        score += 1
    elif len(password) <= 11:
        score += 2
    else:
        score += 3
    # rule 2: uppercase and lowercase letters
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    if has_upper and has_lower:
        score += 3
    else:
        score += 1
    # rule 3: digits
    num_digits = len(re.findall(r'\d', password))
    if num_digits <= 1:
        score += 1
    else:
        score += 2
    # rule 4: special characters
    num_special_chars = len(re.findall(r'[!@#$%^&*()]', password))
    if num_special_chars <= 1:
        score += 1
    else:
        score += 2
    avg_score = score / 4 #calculate the average score

    # determine password classification
    if avg_score <= 1.5:
        return "Weak"
    elif avg_score <= 2.5:
        return "Medium"
    else:
        return "Strong"

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

