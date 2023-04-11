import sys
from feature_extraction import extract_features

if __name__ == "__main__":
    l = len(sys.argv)
    
    #Error checking
    try: #Check to see if the first argument is valid
        inputtest = sys.argv[1]
        file = open('data/' + inputtest)
    except IndexError: #raises if an argument was not given
        print('The first argument states the name of the CSV file with the unhashed passwords')
        sys.exit(1)
    except FileNotFoundError: #raises if the file does not exist
        print("The file does not exist. ")
        sys.exit(1)
        
    data = 'data/' + sys.argv[1] #location of unhashed passwords dataset  
    
    extract_features(data) #function that generates the dataset
