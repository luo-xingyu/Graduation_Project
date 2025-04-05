from detectgpt import GPT2PPLV2 as GPT2PPL
model = GPT2PPL()
sentence = '''We present the first dense SLAM system capable of reconstructing non-rigidly deforming scenes in real-time, by
fusing together RGBD scans captured from commodity sensors. Our DynamicFusion approach reconstructs scene geometry whilst simultaneously estimating a dense volumetric 6D motion field that warps the estimated geometry into
a live frame. Like KinectFusion, our system produces increasingly denoised, detailed, and complete reconstructions
as more measurements are fused, and displays the updated
model in real time. Because we do not require a template
or other prior scene model, the approach is applicable to a
wide range of moving objects and scenes.
3D scanning traditionally involves separate capture and
off-line processing phases, requiring very careful planning
of the capture to make sure that every surface is covered. In practice, it’s very difficult to avoid holes, requiring several iterations of capture, reconstruction, identifying
holes, and recapturing missing regions to ensure a complete
model. Real-time 3D reconstruction systems like KinectFusion [18, 10] represent a major advance, by providing users
the ability to instantly see the reconstruction and identify
regions that remain to be scanned. KinectFusion spurred a
flurry of follow up research aimed at robustifying the tracking [9, 32] and expanding its spatial mapping capabilities to
larger environments [22, 19, 34, 31, 9].
However, as with all traditional SLAM and dense reconstruction systems, the most basic assumption behind
KinectFusion is that the observed scene is largely static.
The core question we tackle in this paper is: How can we
generalise KinectFusion to reconstruct and track dynamic,
non-rigid scenes in real-time? To that end, we introduce
DynamicFusion, an approach based on solving for a volumetric flow field that transforms the state of the scene at
each time instant into a fixed, canonical frame. In the case
of a moving person, for example, this transformation undoes the person’s motion, warping each body configuration
into the pose of the first frame. Following these warps, the
scene is effectively rigid, and standard KinectFusion updates can be used to obtain a high quality, denoised reconstruction. This progressively denoised reconstruction can
then be transformed back into the live frame using the inverse map; each point in the canonical frame is transformed
to its location in the live frame (see Figure 1).
Defining a canonical “rigid” space for a dynamically
moving scene is not straightforward. A key contribution
of our work is an approach for non-rigid transformation and
fusion that retains the optimality properties of volumetric
scan fusion [5], developed originally for rigid scenes. The
main insight is that undoing the scene motion to enable fusion of all observations into a single fixed frame can be
achieved efficiently by computing the inverse map alone.
Under this transformation, each canonical point projects
along a line of sight in the live camera frame. Since the
optimality arguments of [5] (developed for rigid scenes) depend only on lines of sight, we can generalize their optimality results to the non-rigid case.
Our second key contribution is to represent this volumetric warp efficiently, and compute it in real time. Indeed,
even a relatively low resolution, 2563 deformation volume
would require 100 million transformation variables to be
computed at frame-rate. Our solution depends on a combination of adaptive, sparse, hierarchical volumetric basis
functions, and innovative algorithmic work to ensure a realtime solution on commodity hardware. As a result, DynamicFusion is the first system capable of real-time dense reconstruction in dynamic scenes using a single depth camera.
The remainder of this paper is structured as follows. After discussing related work, we present an overview of DynamicFusion in Section 2 and provide technical details in
Section 3. We provide experimental results in Section 4 and
conclude in Section 5
'''
length = len(sentence)
if length < 300:
    chunk_value = 50
elif length < 800:
    chunk_value = length // 6
elif length < 1500:
    chunk_value = length // 10
else:
    chunk_value = 150
print(model(sentence, 250, "v1.1"))
print(len(sentence),chunk_value)    

   