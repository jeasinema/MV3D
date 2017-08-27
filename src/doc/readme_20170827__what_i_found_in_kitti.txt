test for fuse_img_front_1 and fuse_only_1:
1. for most of the case, it turns out that the lookup area should be expaneded
	- for the fat object
	- for the proposal being trunced since it reach the bounding(75,77), so the input should be large enough
2. for like49,50,55, fo1 better than fif1, which turns out that the image and front view feature extractor needed fine training (maybe not,see 63, 68)
3. for 52, it seems that it's hard to detect car whose ry is not 0/90
**4. for 47, especially the car in the distance, RPN do generate porposal for it, but 2nd stage truncs it, which implies that out input imformation is not enough or some representation or encode method for input is not propoer.
And another problem is, the penalty for missing a positive sample is just a classfication loss, which may be not enough. -> add weight on loss for this case!!!
- Such problem also occurs in RPN, so add more weight for this case in RPN, too.
- Look at 264, due to this problem in RPN, we cannot generate enough proposal for some targets, so they are missed in 2nd stage.

5. you shoud trunc some really small bbox result to preventing FP, like 37
6. maybe the image feature still take a large part in weight, like 84, when the image feature is bad, the result will be bad in the same time.

7. generate proposal from top view has some inherent problem, like 95,157,191,236, the car being masked be another car can be seen in rgb but cannot be seen in lidar since the laser cannot reach there.
	-> for 191, it seems that the problem is more about the score is not high enough, may be we should make the penalty for positive sample score w.r.t. the threshold in NMS. 

8. FO1 is better than FIF1 especially when cars are fully occluded(117, 146)

*9. occluded cars which are stoped vertically are easier to deteced than splitlly(146, 149), and the detection for vertically stoped car is really stable!
	-> So why?	
		maybe because: dense lidar feature -> rich proposal
			       dense lidar feature and enough rgb-> good 2nd stage regression

*10. when detect close object, FIF1 is much better than FO1,(146, 152), but for far object, FO1 is much better(159)
-> No, there is nothing to do with far/near(like 176, 185), it just because FO1's feature extractor is much better than FIF1, so we must take care of this -> seperately train a feature extractor!!!! iter more than 300K !!!
-> seperately train every module!!!!

*11. like 244, sometimes, FO1 will generate small bbox for near object
	-> for 463, if we lower the threshold then we can get small box for some missed object, it just imply that the model is not trained well. So small boxes are just a signal that the boxes are not trained well.

*12. like 246, 248 and especially 249, +264,+386+470, if a car do have pattern on BV but not have evident pattern at rgb(like highly occluded), it is easilly to be trunced
	-> but here comes another problem is, how such 2d detector detects such fully-occluded object without extra input?
	-> refer to RRC(currently ranked first in KITTI 2d car hard), it can be done by using context aware refinement, like multi-scale feature and the feature aroud the object, RoI pooling is somehow a kind of this method.
Since it is impossible to generate pattern from zero, so the pattern between different input in fusion net should be supplied for each other in a more reasonalbe manner!!!!!, instead of compete between each other.
	-> for some extreme case(285), it seems that:
	- RGB seems dominant the 2nd stage detector but only tends to miss some positive sample
	- proposals provided in RPN for 2nd stage are not enough for real object, it seems that if there is only 1 proposal for an object, it is always being trunced in the 2nd stage detector. Just look at 285, there are 10 objects in a scan, but the proposal is not enough for some object
	- maybe we must add some severe penalty for RPN and 2nd stage if they give low score for postive sample. and explicitly control the amount of proposal for 2nd stage detector during training.

	-> for the case when the info on BV and RGB are both sparse(highly occluded), what can we do(470, when we lower the threshold, it can be seen that the regress for bbox are also very bad)?
+ 488
		- maybe we should refer to 2d detection:
		1 introduct context aware refinement for 3d detection
		2 increase the training iterations.
	
12+. 264, 274, 275, 323(even near vertically), car which is not stopped verticaly or splitlly, are easily trunced even there are rich BV pattern. compared to 284, vertically and spiltlly are ok
	-> extreme case(360, 425, 432, 441, and all the track like 430), it seems that it is really hard to turn the angle of anchors, even the object is easy to detect in fact.
	-> for 432,445, why vertical/splittly and even squre proposal cannot pooling the feature well for the object which is not stopped vertically and splittly? compare to 2d detection?
	- I think that may partly because the BV pattern is very sparse.
	-> for 441, 432, even RPN can provided enough vertical/spilt proposals for such object, it is also easilly being trunced.
	-> for 442, it seems that only when the rgb or FV is good, such object can be detected well.
	-> So why?
		from rule16, we see that the proposal generation has nothing to do with turning, so this can be only due to the 2nd stage regress capability.

13. like 262, FO1 tends generate small boxes(compared to FIF1?)
	-> no, look at 310, FIF1 also has the same problem, so that may just because the model is not trained well

14. 310,311,316, BV does suplies the RGB for rgb-fully-occluded object, but when an object are both fully occluded(296), it is impossible for detection
-> see 215, which is a typical one for detection under the lack-of-information
(RRC paper, a van which is partly occluded by a wall)

15. for the traffice jam in highway track, such 357, why the RPN tends to generate few proposal for something seems more evident in BV?
	-> in fact, I think this is just "you think" it is rich in BV feature, but from the perspective of RPN, it's not the case.

*16. compared 462 and 357. in 357, a vertical car which BV pattern is evident(But not pretty evident) has only few proposals with low score, so plus the RGB occlude, it's being trunced, in 462, although the car is neither vertical nor split, there are still many proposal here, but in 2nd, it seems that the detector cannot turn the dir of the bbox so the detection result are pretty bad.
	-> Why cause the object in 357 only has few proposals?
		- that may partly due to the sparse pattern, nothing to do with single car, if a single car is near the lidar(which means its pattern is dense enough) it is easy to be deteced(if stopped vertically or splitlly).	
	-> Why cause the object in 462 has many proposals?
		- since RPN is a CNN, it is not sensetive to turning, so the proposal cannbe generated well if the BV pattern is dense. 
	-> Seems enough proposals are the fundermetal for stable detection.

17. (475, 497)for FO1, sometimes it may generate small box for so-called easy object

18. when iou is 0.7, the height of bbox matters, but it seems that it is hard for 2nd stage detector to do it.

19. Why in such as 291, the RPN generate so many proposals at the road side?? Although it has very limited effect on 2nd stage detection.
	- maybe this is because the amount of sample training RPN is not enough, so the negative are not controlled well.


