# Script Rev Eng

### Issues
- [2]can't generate a report from a python script (problem of exe path for the lib pandoc, see incident folder)

### Current features
- automl h2o / (h2orandomforestestimator commented) / h2o metrics 
- wrapper h2o for LIME / LIME metric 
- genetic programming (DEAP), can still be more tuned.
- calculating outliers / displaying outliers found in clusters.
- selecting epsilon for dbscan regarding the distance neighbors plot.
- dbscan (numerical variables + frequency values for categorical variables)
- generic program. It takes as input a dataset (fill the config.json to specify informations about the file)
- principales commandes shell: q, codb, gp, automl, cl

### In development
- test package shap [https://github.com/SeanPLeary/shapley-values-h2o-example](https://github.com/SeanPLeary/shapley-values-h2o-example) / [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- test hdbscan
- add to the q command, a feature to kill processes.
- add a case to switch between 'regression' / 'classification'

#### Warning
-Objets Model H2O, method partial_plot not implemented in version 3.28.2 (check if a depreciated version will make the method work)
<br/>- big datasets (68mo file) don't work, a request for a more powerful computer is in the pipe. (Generic Loan Request 82211)
Job with key $03017f00000132d4ffffffff$_973a3b962ad7f0ac902ab994fab4dbd failed with an exception: DistributedException from /127.0.0.1:54321: 'Java heap space', 
caused by java.lang.OutOfMemoryError: [...] \r\n\tat jsr166y.ForkJoinWorkerThread.run(ForkJoinWorkerThread.java:104)\r\nCaused by: java.lang.OutOfMemoryError: Java heap space\r\n
<br/>-When reenter the command gp-> will display warning. Overwritten methods because declared in a function not in global scope. [https://github.com/DEAP/deap/issues/108](https://github.com/DEAP/deap/issues/108)

### Documentation
#### Test on a new file
Fill the `config.json` as below
```json
{
"yourfile": {
		"path": "/data/yourfile.csv",
		"target_output": "your_output",
		"exclude": ["your_ouput", "field_exclude"],
		"categorical_features" : [0,1,4],
		"inputing_features_gp": [3,13,14]
	}
}
```
Always include the target output in the `exclude` key: 
```
    "exclude": ["your_ouput", "field_exclude"],
```

Add indexes by always counting from `0`:
```json
  "categorical_features" : [0,1,4,5,6,8,9],
  "inputing_features_gp": [3,13,14]
```


