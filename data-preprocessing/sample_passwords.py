import random
import csv

def sample_passwords(data_filepath, sample_size):
    with open(data_filepath, 'r', encoding='ISO-8859-1') as file: # open the file
        passwords = file.readlines() # read it
    sampled_passwords = random.sample(passwords, sample_size) # randomly subset the passwords
    output_file = data_filepath[:-4] + "_subset.csv" # create the name of the output
    with open(output_file, 'w', newline='') as csvfile: # open the output
        writer = csv.writer(csvfile) # open writer
        writer.writerow(['password']) # write the header
        for password in sampled_passwords: # iterate through each of the sampled passwords
            writer.writerow([password.strip()]) # write to the csv file
    print("We have subsetted the data")
