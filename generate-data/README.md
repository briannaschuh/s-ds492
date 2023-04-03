./GenerateHashes {name of file with passwords} {name of the file you would like to generate with the hashes}

This program takes .txt or .csv file that contains unhashed passwords and outputs a csv file that has the unhashed passwords, their sha1 encryptions, and their sha256 encryptions.

The first argument is the name of the file that contains the passwords, and the second file is the name of the file that you would like to output. 

The output can be fed into the LSTM model, the Seq2Seq model, or the CNN model provided on this repo. 
