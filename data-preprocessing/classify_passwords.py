import sys
from feature_extraction import extract_features
from sample_passwords import sample_passwords
from classify_passwords import classify_passwords

if __name__ == "__main__":
    l = len(sys.argv)
    
    
    # error checking
    try: # check to see if the first argument is valid
        inputtest = sys.argv[1]
    except IndexError: # raises if an argument was not given
        print("The first argument must specify the action. ")
        sys.exit(1)
    
    try: # check to see if the second argument is valid
        inputtest = sys.argv[2]
        file = open('data/' + inputtest)
    except IndexError: # raises if an argument was not given
        print('The second argument states the name of the CSV file with the unhashed passwords')
        sys.exit(1)
    except FileNotFoundError: # raises if the file does not exist
        print("The file does not exist. ")
        sys.exit(1)
        
    command = sys.argv[1]
    data = 'data/' + sys.argv[2] # location of the dataset
    
    # check to see if the command is valid
    if command != 'extract' and command != 'subset' and command != 'classify':
        print("Please use one of the valid commands: extract, subset, or classify")
        sys.exit(1)
    
    # check to see if the third argument was given if the commad is subset
    if command == 'subset':
        try:
            inputtest = sys.argv[3]
        except IndexError:
            print('The third argument for this command states the size of the subset')
            sys.exit(1)
    
    if command == 'extract':
        extract_features(data) # function that extracts the features
    if command == 'subset':
        sample_passwords(data, int(sys.argv[3])) # function that subsets the data
    if command == 'classify':
        classify_passwords(data) # fuction that classifies the password
