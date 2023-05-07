This a program that pre-processes the data so that it can be inputted directly into the supervised and unsupervised models. 

# Examples of how to pre-process the data:

## Subset a dataset

./Preprocess subset file_name sampl_size 

./Preprocess subset 000webhost.txt 700000

## Classify password strengths

./Preprocess classify file_name

./Proprocess classify 000webhost_subset.csv

## Extract features

./Preprocess extract file_name

./Preprocess extract 000webhost_subset_classified.csv
