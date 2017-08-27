So how about the fix?
0. enrich training set
1. fix the lookup region, extend it to ...
2. add extra penalty for behaviors like giving a positive sample a low score, both RPN and 2nd stage
	-> to see if we can handle the problem in 264, so as to provide more proposals for target objects
	-> the penalty should be made w.r.t. NMS's threshold. 
3. seperately trained a RPN and a feature extractor first for more interation, then use them to train a fuse net
4. explicitly control the sample amount for 2nd stage during training stage
	-> to see if giving more proposal and set the severe penalty, the bv and rgb can reduce the missing TP(true positive) amount
5. redesign the fusion structure, to explicitly make the rgd, front, lidar supply for each other instead of compete -> you can make a balance for compete and supply using some novel mechanism
6. for introduce context aware information for BV, since it is extremely sparse, how can we do it?
7. design some structure to make the proposal turnable, since the 2nd cannot turn the angle of proposal well
8. increase the training iterations.
9. to design some structure for height regression
10. pay attention to the sample amount, loss weight, pos/neg ratio when you enlarge the size of the RPN input. 
