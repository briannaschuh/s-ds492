import random
import csv

def sample_passwords(data_filepath, sample_size):
    allowed_chars = "!#$%&'()*+-./0123456789:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¡"

    with open(data_filepath, 'r', encoding='ISO-8859-1') as file:  # open the file
        passwords = file.readlines()  # read it

    # Filter passwords containing only allowed characters
    filtered_passwords = [password.strip() for password in passwords if all(char in allowed_chars for char in password.strip())]

    # Randomly subset the filtered passwords
    sampled_passwords = random.sample(filtered_passwords, sample_size)

    output_file = data_filepath[:-4] + "_subset.csv"  # create the name of the output

    with open(output_file, 'w', newline='') as csvfile:  # open the output
        writer = csv.writer(csvfile)  # open writer
        writer.writerow(['password'])  # write the header
        for password in sampled_passwords:  # iterate through each of the sampled passwords
            writer.writerow([password])  # write to the csv file

    print("We have subsetted the data")


