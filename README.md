# HDDM_extensions
some usefull functions for HDDM(hierarchical drift diffusion model)

## In Concurrent_HDDM
I try several ways to run multitask for HDDM, including joblib and pathos.

The codes of single task with multi-parameter and multitask with multi-parameter are encapsulated. Thus, you could easily recall the two funcitons to constructing your own models.

Note, Our example suggest some errors(models with exactly same results) would happen when using different python package. So, I recommend adding the dbname argment in your HDDM.sample() func to avoid above error.

