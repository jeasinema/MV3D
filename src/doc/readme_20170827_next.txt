So how about the fix?
0. enrich training set  OK

1. fix the lookup region, extend it to ...   OK

2. add extra penalty for behaviors like giving a positive sample a low score, both RPN and 2nd stage  OK
	-> to see if we can handle the problem in 264, so as to provide more proposals for target objects
	-> the penalty should be made w.r.t. NMS's threshold. 

3. seperately trained a RPN and a feature extractor first for more interation, then use them to train a fuse net

4. explicitly control the sample amount for 2nd stage during training stage  OK
	-> to see if giving more proposal and set the severe penalty, the bv and rgb can reduce the missing TP(true positive) amount

5. redesign the fusion structure, to explicitly make the rgd, front, lidar supply for each other instead of compete -> you can make a balance for compete and supply using some novel mechanism
(in fact, refer to the result of MV3D, it seems that MV3D == BFV3D, rgb takes a little part. so improve your FV is very important. 
 -> STILL NEED SOME FIX ON LOSS  OK 

6. for introduce context aware information for BV, since it is extremely sparse, how can we do it?
-> Maybe the only thing I can do now is to enlarge the bbox   OK
-> Maybe you should take a look at the feature embedding of BV 

***7. design some structure to make the proposal turnable, since the 2nd cannot turn the angle of proposal well
-> In fact, I think this maybe partly due to the anchors design, you can just draw and cal, then you will find out that for 45* tilt object, there is no anchor can be treated as positive samples, which means that the RPN do not learn how to regress to such object.
-> after some exp, I found this idea is not correct, since although we shoud improve the anchor design, for such object, RPN can still generate enough proposals. And during the exp, I've also found that the the with_front_8 version do not have the problem of generating proposal with high belief in the side. -> this is may because that for the fusion_img_front, I set the cls loss weight for negative sample to 0.05, which is small and cannot limit it.
so add more weight for negative one or increase the training iterations.
-> A naive method may be first train a BV detector, then stack some height regress layer on it, but such method is not end2end.


8. increase the training iterations.

***9. to design some structure for height regression

10. pay attention to the sample amount, loss weight, pos/neg ratio when you enlarge the size of the RPN input. 

11. since the lidar input is very sparse, you cannot require the RPN to give you accurate objectness result of each anchors, so we must lower the threshold of NMS in RPN stage  see 12, OK

12. just remove the RPN NMS during training, since if you do not train the RPN and the RCNN simultaneously, or so-called you train RPN first, there maybe noting can be regarded as negative sample in 2nd stage detector or just non-sense sample like the anchors in the road side  OK
>>> https://arxiv.org/pdf/1702.02138.pdf 
Accoring to this note, NMS in Faster-RCNN is a tool for hard-sample mining, which can bias the training to small object, then speed up the training process. 
That is because big proposals have large overlaps, so they're more easily to be supressed in the NMS step. 
The author prove this by explicitly using the scale of proposal as a sample ratio, and found such method can finally catch up the NMS version if trained with enough iterations.
For another case, the best setting is training with NMS and testing without NMS(Just the top-K proposal to feed in the 2nd detector).
-> And this note also give some reasonable parameter:
- RPN_TRAIN_BATCH 256
- RCNN_TRAIN_BATCH 128, batch size == 1
- pos/neg ratio: 1:4
- RPN NMS pre top-K: 12000
- RPN NMS post top-K: 2000
- RPN NMS threshold when training: 0.7





13. add decov layer for the top view before proposal generation OK

14. change the way you encode the front, now it seems that only 3.74% point is remained.....(most of the point occluded..)  OK
-> Just maintain the 1500*100 instead cut it down by 2x
-> And it can be seen that the front view is very dense.

15. For the RPN/RCNN param(configuration.py):  OK
[+] RPN_BATCHSIZE: because we want to supress the fp in RPN, increase this can make the fp easier to be generate loss for training RPN
[0] RPN_FG/BG_THRESH
[+] RPN_NMS_PRE/POST_TOPN: this is origin reason why we got very few proposal in 2nd stage during the training precedure
[0] RCNN_BATCH_SIZE: this is just because 128 is what cxz mentioned in his paper
[+/-] RCNN_BG_THRESH: because we use it now in the fuse_target
[0] RCNN_FG_THRESH: have no idea how to fine-tune it, so just remains it unchanged

16. add extra weight of the point amount in a bbox, to prevent the model from overriding the far/occluded vehicles
(in fact, this is a somehow hard sample mining)    NO_NEED

17. ATTENTION! For RPN training, it just regresses to a 0/90 oriented "large" bbox instead of a tilting one, but for training the 2nd stage, it is ok
-> This will cause a problem that, for rpn_target, more anchors will be regarded as the postive sample for the tilting gt box, since it is tilting, its gt box is larger for cal iou.

18. So it seems that if set the FG/BG_IOU_THRESHOLD for RPN sample selection to 0.5/0.3, faster-rcnn's anchors

19. Start Training!
-> RPN 
-> 

20. consider add a truncer to remove empty anchors(using inside_inds)


!!! batch size rewrite! OK
-> 20170905
e.g.
top_cls_loss -> to compatible with original version, this is the target for optimize
top_cls_loss_sum -> cumulation of loss, to feed in graph, equal to top_cls_loss
top_cls_loss_cur -> immediate val of loss in cur iteration, mainly for cumulation and scalar log 

-> for some optimizer, the method you handle nan(just set it to 0) may hurt them(consider 2nd derivative or momentum based method), so the proper way is to just do not do optimize this time
but for common 1st derivative based optimizer, it is ok to do that, equal to do not generat delta for param this time.
-> 20170905 the question above is still under reviewed

!!! single rpn mode!!!  OK 
!!! multiprocessing loading  OK
20170904 19:00
1. add the weight for negative cls loss in RPN
2. using real batch size == 1
3. single rpn mode, speed up loading
4. using 0.5/0.7 instead of 0.3/0.5 for pos/neg in rpn stage

20170905 18:00
1. set batch size to 16/8, weight fixed at 1
2.shuffle the dataset
it seems that batch size is ok, but the model cannot continue to converge when large batch size... but it do converge faster
so set it to 2/4

20170906 12:00
1. reset the batch size to 1 -> a train
2. set reg loss to 0.05 -> a train  
	settings with 1+2 -> result is bad...reg cannot converge at 150k

It seems that after enlarge the input size, all the result do not converge well.... and large batch size is really bad for kitti...


20170906 19:00
1. rpn with batch size == 1 weight == 0.05
	-> result is bad...reg cannot converge at 150k
2. rpn with batch size  == 8/16 lr == 0.008/0.016 weight =0.5
	-> cannot converge as fast as batch size == 1
 now all the batch size > 1 cannot converget the regress well
damnit! all bad! reg loss not coverge!!!

20170907 22:30
1. removal (thresh=0), batch size 1, 0.05 rpn pos=2
2. removal batch size 1 weight 0.5 rpn pos=2
3. removal batch size 1  weight 1  rpn pos=2
4. removal batch size 4 weigth 0.5 lr=0.004 rpn pos=2
result are bad, not converge well

20170912 
both remove no gt 
1. fixed at batch size 1, removal(thresh=0), 0.05, rpn pos=2
2. fixed at batch size 1, removal(thresh=0), 0.5, rpn pos=2
3. fixed at batch size 1, removal(thresh=0), 1, rpn pos=2
seems that the dataset is very dirty(many without gt...)
