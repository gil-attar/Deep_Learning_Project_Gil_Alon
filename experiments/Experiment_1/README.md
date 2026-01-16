To run experiment one: \
call ./run_experiment1.sh FROM INSIDE THIS FOLDER. 

this is a bash script that runs all 8 runs (2 models, 4 runs per model F0-F4) by calling
runOneTest.py.
If it one of the runs fail, it tries it again. if the second attempt fails, it moves over
to run the next run.

## Protocol freeze (important)
Experiment 1 follows a frozen run/evaluation contract to ensure comparability and evaluator compatibility.

- Contract file: `experiments/Experiment_1/eval_contract.json`
- README form: `experiments/Experiment_1/RUN_CONTRACT.md`

Do not regenerate split/evaluation index files or change evaluation thresholds mid-experiment unless the dataset changes and all configurations are re-run.
