Code for obtaining results described in the following paper:

> Grimm R., Cassani G., Gillis S. and Daelemans W. (2017). Evidence for a facilitatory effect of multi-word units on child word learning. Proceedings of the 39th annual conference of the cognitive science society.

## OS and Dependencies

This project is written in Python (version 3.4.3) and R (version 3.3.3), both on Ubuntu 14.04. The biggest part of the code is written in Python, and a small part for statistical analysis is written in R. 
The Python component requires the following packages (the version we used is given in parentheses):
> numpy (1.12.1)  
nltk (3.2.2)  
scipy (0.19.0)  

## Get the Corpus Data

#### Prepare the CHILDES corpora 

We use several corpora from the [CHILDES data base](http://childes.talkbank.org/).  

Get the North American corpora [here](http://childes.talkbank.org/data-xml/Eng-NA/).    
Then unzip them to: Frontiers_MultiWordUnits/CHILDES/corpora/NA/

Then, get the British English corpora [here](http://childes.talkbank.org/data-xml/Eng-UK/).  
Unzip them to: Frontiers_MultiWordUnits/CHILDES/corpora/BE/

Download the following North American corpora:
> Bates, Bernstein, Bliss, Bloom70, Bloom73, Bohannon, Braunwald, Brent, Brown, Carterette, Clark, Cornell, Demetras1, Demetras2, ErvinTripp, Evans, Feldman, Garvey, Gathercole,  Gleason, HSLLD, Hall, Higginson, Kuczaj, MacWhinney, McCune, McMillan, Morisset, Nelson, NewEngland, Peters, Post, Providence, Rollins, Sachs, Snow, Soderstrom, Sprott, Suppes, Tardif, Valian, VanHouten, VanKleeck, Warren, Weist

And the following British English corpora:  
> Belfast, Fletcher, Manchester, Thomas, Tommerdahl, Wells, Forrester, Lara

## Run the experiments 

The project's root directory contains Python and R scripts, numbered 1 through 6, which you need to run one after the other in order to carry out the experiments. 

*1-pre_process_cds_corpora.py*      
Pre-process the CHILDES corpora.

*2-induce_aofp.py*       
Collect age of first production (AoFP) values for words used by the children in the CHILDES corpora.

*3-run_chunk_based_learner.py*        
Run the Chunk-Based Learner on the on the CHILDES corpora and save extracted multi-word units to hard drive.

*4-run_chunk_based_learner.py*        
Run the Prediction Based Segmenter on the on the CHILDES corpora and save extracted multi-word units to hard drive.  
Code for the Prediction Based Segmenter was written by Julian Brooke and is taken from his homepage at the University of Toronto: http://www.cs.toronto.edu/~jbrooke/ngram_decomp_seg.py  
We converted the program to to Python3 via [2-to-3](https://docs.python.org/2/library/2to3.html) and made minor changes to integrate it into our pipeline.  

*5-results_to_csv.py*  
Compute statistics and write results to CSV file for statistical analysis in R.

*6-statistical_analysis.R*  
Perform statistical analysis in R.

To get the results for various tables, check the scripts in: ./Tables
