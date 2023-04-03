from passlib.hash import ldap_sha1, sha256_crypt
import os

def generate_hashes(input_file, output_file):
    #open the file with the passwords
    with open(input_file, 'r+', encoding = "ISO-8859-1") as input, open(output_file, 'w') as output:
        #set the headers for the csv file that will be outputted
        output.write("Unhashed password, SHA1, SHA256\n")
        for line in input: #itereate through each line in the passwords file
            line = line.strip() #strip the line so we just get the characters
            sha1_hash = ldap_sha1.hash(line) #hash using sha1
            sha256_hash = sha256_crypt.hash(line) #hash using sha256

            new_line = f'{line}, {sha1_hash[5:]}, {sha256_hash[17:]}\n' #line that will go into the output
            output.write(new_line) #append the new line to the output file
            output.flush() #save the file
            os.fsync(output.fileno())
            
