------------------------------------------------------------------------------------------------------------------------------------
PIO-Gen
------------------------------------------------------------------------------------------------------------------------------------


**University of Colorado at Colorado Springs**

----------------------------------------------------------------------

**Developer:** <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sean Higgins <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado at Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: shiggins@uccs.edu

	 
--------------------------------------------------------------------	

**1.	Content of folders:**
-----------------------------------------------------------	
* **PIO-Gen.py**: Driver for pigeon inspired search
* workers.py: Holds the definition of the members of the swarm 
* output: example outputs
* data: example inputs

**2.	Input matrix file format:**
-----------------------------------------------------------
Requires:
* functools
* numpy
* scipy
* sklearn

**3.	Input matrix file format:**
-----------------------------------------------------------
PIO-Gen allows for one format:
* Square Matrix Input format: The square matrix is a tab seperated N by N contact matrix derived.


**4.	Usage:**
-----------------------------------------------------------
**4.1. 	Python:** <br />
To run the tool 	 **python PIO-Gen.py <input_path> <output_path> <conversion_factor> <cluster_size>** 

- Parameters:
	+ input_path: path to input matrix
	+ output_path: path to output (Do not use an extention they are automatically added)
	+ conversion_factor: Desired conversion facter. Values between 0.1 and .4 have tested the best.
	+ cluster_size: the number of bins that are compaired when searching for optimum 3d shape. Must be a multiple of the matrix size.

Example:
python PIO-Gen.py Data/regular70.txt output/test 0.4 10

**5.	Output**
-----------------------------------------------------------
PIO-Gen produces 2  files

- Output:
	+ *.pdb: cordinates for predicted structure
	+ *.log: contains conversion factor and average root means square error (RMSE) and average correlation of Spearman's and Pearson's correlations

