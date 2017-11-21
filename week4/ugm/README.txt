Instructions to run the code
=============================

- Images:

	- 3_12_s.bmp: house
	- 2_1_s.bmp: tree
	- 7_9_s.bmp: car

- Main algorithm:

	-Specifications:
		- Potts Model from assigned values
		- Color clustering, Loopy Belief Propagation and Max Sum
	
	- Instructions:
		- Run:
		
		UGM_segmentation_X
		
		where X is one of house, tree or car
		
- Optional 1:

	-Specifications:
		- Potts Model from Boltzmann distribution
		- Color clustering, Loopy Belief Propagation and Max Sum
		- House image (3_12_s.bmp)
	
	- Instructions:
		- Run:
		
		UGM_segmentation_Boltzmanndistr
		
- Optional 2.1:

	-Specifications:
		- Potts Model from assigned values
		- Color clustering, Loopy Belief Propagation, Max Sum, Graph Cuts
	
	- Instructions:
		- Run:
		
		UGM_segmentation_X_GC
		
		where X is one of house, tree or car
		
- Optional 2.2:

	-Specifications:
		- Potts Model from assigned values
		- Color clustering, Loopy Belief Propagation, Max Sum, Iterated Conditional Modes
	
	- Instructions:
		- Run:
		
		UGM_segmentation_X_ICM
		
		where X is one of house, tree or car
		
- Notes: To run Graph Cuts we used the maxflow library to speed up its execution:

	https://es.mathworks.com/matlabcentral/fileexchange/21310-maxflow