import sys
from generate_hashes import generate_hashes

if __name__ == "__main__":
    l = len(sys.argv)
    
    #Error checking
    try: #Check to see if the first argument is valid
        inputtest = sys.argv[1]
        file = open('unhashed_passwords/' + inputtest)
    except IndexError: #raises if an argument was not given
        print('The first argument states the name of the CSV file with the unhashed passwords')
        sys.exit(1)
    except FileNotFoundError: #raises if the file does not exist
        print("The file does not exist. ")
        sys.exit(1)
        
    try: #check to see if the second argument is valid
        inputtest = sys.argv[2]
    except IndexError: #raises if the second argument was not given
        print('The second argument states the name of the CSV file you would like to create with the hashed passwords. ')
        
    unhashed = 'unhashed_passwords/' + sys.argv[1] #location of unhashed passwords dataset
    output = sys.argv[2] #name of the new dataset that will be generated     
    
    generate_hashes(unhashed, output) #function that generates the dataset
