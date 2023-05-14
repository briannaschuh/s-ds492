This a program that preprocesses the data so that it can be inputted directly into the supervised and unsupervised models. 

# Examples of how to preprocess the data:

## Subset a dataset

./Preprocess subset file_name sample_size 

./Preprocess subset 000webhost.txt 700000

## Classify password strengths

./Preprocess classify file_name

./Proprocess classify 000webhost_subset.csv

## Extract features

./Preprocess extract file_name

./Preprocess extract 000webhost_subset_classified.csv
