To run experiment one: \
call ./run_experiment1.sh FROM INSIDE THIS FOLDER. 

this is a bash script that runs all 8 runs (2 models, 4 runs per model F0-F4) by calling
runOneTest.py.
If it one of the runs fail, it tries it again. if the second attempt fails, it moves over
to run the next run.

