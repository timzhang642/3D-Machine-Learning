# 3D Machine Learning
In recent years, tremendous amount of progress is being made in the field of 3D Machine Learning, which is an interdisciplinary field that fuses computer vision, computer graphics and machine learning. This repo is derived from my study notes and will be used as a place for triaging new research papers. 

I'll use the following icons to differentiate 3D representations:
* :camera: Multi-view Images
* :space_invader: Volumetric
* :game_die: Point Cloud
* :gem: Polygonal Mesh
* :pill: Primitive-based

## Get Involved
To make it a collaborative project, you may add content throught pull requests or open an issue to let me know. 

## Available Courses
[Stanford CS468: Machine Learning for 3D Data (Spring 2017)](http://graphics.stanford.edu/courses/cs468-17-spring/)

[MIT 6.838: Shape Analysis (Spring 2017)](http://groups.csail.mit.edu/gdpgroup/6838_spring_2017.html)

[Princeton COS 526: Advanced Computer Graphics  (Fall 2010)](https://www.cs.princeton.edu/courses/archive/fall10/cos526/syllabus.php)

[Princeton CS597: Geometric Modeling and Analysis (Fall 2003)](https://www.cs.princeton.edu/courses/archive/fall03/cs597D/)

[Geometric Deep Learning](http://geometricdeeplearning.com/)

## Datasets
To see a survey of RGBD datasets, I recommend to check out Michael Firman's [collection](http://www0.cs.ucl.ac.uk/staff/M.Firman//RGBDdatasets/) as well as the associated paper, [RGBD Datasets: Past, Present and Future](https://arxiv.org/pdf/1604.00999.pdf). Point Cloud Library also has a good dataset [catalogue](http://pointclouds.org/media/). 

## Single Object Classification
:space_invader: <b>3D ShapeNets: A Deep Representation for Volumetric Shapes (2015)</b> [[Paper]](http://3dshapenets.cs.princeton.edu/)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/3ed23386284a5639cb3e8baaecf496caa766e335/1-Figure1-1.png" /></p>

:space_invader: <b>VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition (2015)</b> [[Paper]](http://www.dimatura.net/publications/voxnet_maturana_scherer_iros15.pdf) [[Code]](https://github.com/dimatura/voxnet)
<p align="center"><img width="50%" src="http://www.dimatura.net/research/voxnet/car_voxnet_side.png" /></p>

:camera: <b>Multi-view Convolutional Neural Networks  for 3D Shape Recognition (2015)</b> [[Paper]](http://vis-www.cs.umass.edu/mvcnn/)
<p align="center"><img width="50%" src="http://vis-www.cs.umass.edu/mvcnn/images/mvcnn.png" /></p>

:camera: <b>DeepPano: Deep Panoramic Representation for 3-D Shape Recognition (2015)</b> [[Paper]](http://mclab.eic.hust.edu.cn/UpLoadFiles/Papers/DeepPano_SPL2015.pdf)
<p align="center"><img width="30%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/5a1b5d31905d8cece7b78510f51f3d8bbb063063/1-Figure3-1.png" /></p>

:space_invader::camera: <b>FusionNet: 3D Object Classification Using Multiple Data Representations (2016)</b> [[Paper]](https://stanford.edu/~rezab/papers/fusionnet.pdf)
<p align="center"><img width="30%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/0aab8fbcef1f0a14f5653d170ca36f4e5aae8010/6-Figure5-1.png" /></p>

:space_invader::camera: <b>Volumetric and Multi-View CNNs for Object Classification on 3D Data (2016)</b> [[Paper]](https://arxiv.org/pdf/1604.03265.pdf) [[Code]](https://github.com/charlesq34/3dcnn.torch)
<p align="center"><img width="40%" src="http://graphics.stanford.edu/projects/3dcnn/teaser.jpg" /></p>

:space_invader: <b>Generative and Discriminative Voxel Modeling with Convolutional Neural Networks (2016)</b> [[Paper]](https://arxiv.org/pdf/1608.04236.pdf) [[Code]](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
<p align="center"><img width="50%" src="http://davidstutz.de/wordpress/wp-content/uploads/2017/02/brock_vae.png" /></p>

:space_invader: <b>3D GAN: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (2016)</b> [[Paper]](https://arxiv.org/pdf/1610.07584.pdf)
<p align="center"><img width="50%" src="http://3dgan.csail.mit.edu/images/model.jpg" /></p>

:space_invader: <b>Generative and Discriminative Voxel Modeling with Convolutional Neural Networks (2017)</b> [[Paper]](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
<p align="center"><img width="50%" src="https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling/blob/master/doc/GUI3.png" /></p>

:space_invader: <b>FPNN: Field Probing Neural Networks for 3D Data (2016)</b> [[Paper]](http://yangyanli.github.io/FPNN/)
<p align="center"><img width="30%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/15ca7adccf5cd4dc309cdcaa6328f4c429ead337/1-Figure2-1.png" /></p>

:space_invader: <b>OctNet: Learning Deep 3D Representations at High Resolutions (2017)</b> [[Paper]](https://github.com/griegler/octnet)
<p align="center"><img width="30%" src="https://is.tuebingen.mpg.de/uploads/publication/image/18921/img03.png" /></p>

:space_invader: <b>O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis  (2017)</b> [[Paper]](http://wang-ps.github.io/O-CNN)
<p align="center"><img width="50%" src="http://wang-ps.github.io/O-CNN_files/teaser.png" /></p>

:game_die: <b>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017)</b> [[Paper]](http://stanford.edu/~rqi/pointnet/)
<p align="center"><img width="40%" src="https://web.stanford.edu/~rqi/papers/pointnet.png" /></p>

:game_die: <b>PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.02413.pdf)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PointNet%2B%2B-%20Deep%20Hierarchical%20Feature%20Learning%20on%20Point%20Sets%20in%20a%20Metric%20Space.png" /></p>

:camera: <b>Feedback Networks (2017)</b> [[Paper]](http://feedbacknet.stanford.edu/)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Feedback%20Networks.png" /></p>

## Multiple Objects Detection
<b>Sliding Shapes for 3D Object Detection in Depth Images (2014)</b> [[Paper]](http://slidingshapes.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://slidingshapes.cs.princeton.edu/teaser.jpg" /></p>

<b>Object Detection in 3D Scenes Using CNNs in Multi-view Images (2016)</b> [[Paper]](https://stanford.edu/class/ee367/Winter2016/Qi_Report.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Object%20Detection%20in%203D%20Scenes%20Using%20CNNs%20in%20Multi-view%20Images.png" /></p>

<b>Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images (2016)</b> [[Paper]](http://dss.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://3dvision.princeton.edu/slide/DSS.jpg" /></p>

<b>DeepContext: Context-Encoding Neural Pathways  for 3D Holistic Scene Understanding (2016)</b> [[Paper]](http://deepcontext.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://deepcontext.cs.princeton.edu/teaser.png" /></p>

## Part Segmentation
<b>Learning 3D Mesh Segmentation and Labeling (2010)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/LabelMeshes/LabelMeshes.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/0bf390e2a14f74bcc8838d5fb1c0c4cc60e92eb7/7-Figure7-1.png" /></p>

<b>Unsupervised Co-Segmentation of a Set of Shapes via Descriptor-Space Spectral Clustering (2011)</b> [[Paper]](https://www.cs.sfu.ca/~haoz/pubs/sidi_siga11_coseg.pdf)
<p align="center"><img width="30%" src="http://people.scs.carleton.ca/~olivervankaick/cosegmentation/results6.png" /></p>

<b>3D Shape Segmentation with Projective Convolutional Networks (2017)</b> [[Paper]](http://people.cs.umass.edu/~kalo/papers/shapepfcn/)
<p align="center"><img width="50%" src="http://people.cs.umass.edu/~kalo/papers/shapepfcn/teaser.jpg" /></p>

<b>Learning Hierarchical Shape Segmentation and Labeling from Online Repositories (2017)</b> [[Paper]](http://cs.stanford.edu/~ericyi/project_page/hier_seg/index.html)
<p align="center"><img width="50%" src="http://cs.stanford.edu/~ericyi/project_page/hier_seg/figures/teaser.jpg" /></p>

:game_die: <b>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017)</b> [[Paper]](http://stanford.edu/~rqi/pointnet/)
<p align="center"><img width="40%" src="https://web.stanford.edu/~rqi/papers/pointnet.png" /></p>

:game_die: <b>PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.02413.pdf)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PointNet%2B%2B-%20Deep%20Hierarchical%20Feature%20Learning%20on%20Point%20Sets%20in%20a%20Metric%20Space.png" /></p>

## 3D Synthesis/Reconstruction
_Parametric Morphable Model-based methods_

<b>A Morphable Model For The Synthesis Of 3D Faces (1999)</b> [[Paper]](http://gravis.dmi.unibas.ch/publications/Sigg99/morphmod2.pdf)[[Github]](https://github.com/MichaelMure/3DMM)
<p align="center"><img width="40%" src="http://mblogthumb3.phinf.naver.net/MjAxNzAzMTdfMjcz/MDAxNDg5NzE3MzU0ODI3.9lQioLxwoGmtoIVXX9sbVOzhezoqgKMKiTovBnbUFN0g.sXN5tG4Kohgk7OJEtPnux-mv7OAoXVxxCyo3SGZMc6Yg.PNG.atelierjpro/031717_0222_DataDrivenS4.png?type=w420" /></p>

<b>The Space of Human Body Shapes: Reconstruction and Parameterization from Range Scans (2003)</b> [[Paper]](http://grail.cs.washington.edu/projects/digital-human/pub/allen03space-submit.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/46d39b0e21ae956e4bcb7a789f92be480d45ee12/7-Figure10-1.png" /></p>

_Part-based Template Learning methods_

<b>Modeling by Example (2004)</b> [[Paper]](http://www.cs.princeton.edu/~funk/sig04a.pdf)
<p align="center"><img width="20%" src="http://gfx.cs.princeton.edu/pubs/Funkhouser_2004_MBE/chair.jpg" /></p>

<b>Model Composition from Interchangeable Components (2007)</b> [[Paper]](http://www.cs.princeton.edu/courses/archive/spring11/cos598A/pdfs/Kraevoy07.pdf)
<p align="center"><img width="40%" src="http://www.cs.ubc.ca/labs/imager/tr/2007/Vlad_Shuffler/teaser.jpg" /></p>

<b>Data-Driven Suggestions for Creativity Support in 3D Modeling (2010)</b> [[Paper]](http://vladlen.info/publications/data-driven-suggestions-for-creativity-support-in-3d-modeling/)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2011/12/creativity.png" /></p>

<b>Photo-Inspired Model-Driven 3D Object Modeling (2011)</b> [[Paper]](http://kevinkaixu.net/projects/photo-inspired.html)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/photo-inspired/overview.PNG" /></p>

<b>Probabilistic Reasoning for Assembly-Based 3D Modeling (2011)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/assembly/ProbReasoningShapeModeling.pdf)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2011/12/highlight9.png" /></p>

<b>A Probabilistic Model for Component-Based Shape Synthesis (2012)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/ShapeSynthesis/ShapeSynthesis.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/A%20Probabilistic%20Model%20for%20Component-Based%20Shape%20Synthesis.png" /></p>

<b>Structure Recovery by Part Assembly (2012)</b> [[Paper]](http://cg.cs.tsinghua.edu.cn/StructureRecovery/)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/Structure%20Recovery%20by%20Part%20Assembly.png" /></p>

<b>Fit and Diverse: Set Evolution for Inspiring 3D Shape Galleries (2012)</b> [[Paper]](http://kevinkaixu.net/projects/civil.html)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/civil/teaser.png" /></p>

<b>AttribIt: Content Creation with Semantic Attributes (2013)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/attribit/AttribIt.pdf)
<p align="center"><img width="30%" src="http://gfx.cs.princeton.edu/gfx/pubs/Chaudhuri_2013_ACC/teaser.jpg" /></p>

<b>Learning Part-based Templates from Large Collections of 3D Shapes (2013)</b> [[Paper]](http://shape.cs.princeton.edu/vkcorrs/papers/13_SIGGRAPH_CorrsTmplt.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/Learning%20Part-based%20Templates%20from%20Large%20Collections%20of%203D%20Shapes.png" /></p>

<b>Topology-Varying 3D Shape Creation via Structural Blending (2014)</b> [[Paper]](http://gruvi.cs.sfu.ca/project/topo/)
<p align="center"><img width="50%" src="http://gruvi.cs.sfu.ca/project/topo/teaser.jpg" /></p>

<b>Estimating Image Depth using Shape Collections (2014)</b> [[Paper]](http://vecg.cs.ucl.ac.uk/Projects/SmartGeometry/image_shape_net/imageShapeNet_sigg14.html)
<p align="center"><img width="50%" src="http://vecg.cs.ucl.ac.uk/Projects/SmartGeometry/image_shape_net/paper_docs/pipeline.jpg" /></p>

<b>Single-View Reconstruction via Joint Analysis of Image and Shape Collections (2015)</b> [[Paper]](https://www.cs.utexas.edu/~huangqx/modeling_sig15.pdf)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2015/05/single-view.png" /></p>

<b>Interchangeable Components for Hands-On Assembly Based Modeling (2016)</b> [[Paper]](http://www.cs.umb.edu/~craigyu/papers/handson_low_res.pdf)
<p align="center"><img width="30%" src="https://github.com/timzhang642/test1/blob/master/imgs/Interchangeable%20Components%20for%20Hands-On%20Assembly%20Based%20Modeling.png" /></p>

<b>Shape Completion from a Single RGBD Image (2016)</b> [[Paper]](http://www.kunzhou.net/2016/shapecompletion-tvcg16.pdf)
<p align="center"><img width="40%" src="http://tianjiashao.com/Images/2015/completion.jpg" /></p>

_Deep Learning Methods_

:camera: <b>Learning to Generate Chairs, Tables and Cars with Convolutional Networks (2014)</b> [[Paper]](https://arxiv.org/pdf/1411.5928.pdf)
<p align="center"><img width="50%" src="https://zo7.github.io/img/2016-09-25-generating-faces/chairs-model.png" /></p>

:camera: <b>Weakly-supervised Disentangling with Recurrent Transformations for 3D View Synthesis (2015, NIPS)</b> [[Paper]](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf)
<p align="center"><img width="50%" src="https://github.com/jimeiyang/deepRotator/blob/master/demo_img.png" /></p>

:game_die: <b>Analysis and synthesis of 3D shape families via deep-learned generative models of surfaces (2015)</b> [[Paper]](https://people.cs.umass.edu/~hbhuang/publications/bsm/)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~hbhuang/publications/bsm/bsm_teaser.jpg" /></p>

:camera: <b>Weakly-supervised Disentangling with Recurrent Transformations for 3D View Synthesis (2015)</b> [[Paper]](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf) [[Code]](https://github.com/jimeiyang/deepRotator)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/042993c46294a542946c9c1706b7b22deb1d7c43/2-Figure1-1.png" /></p>

:camera: <b>Multi-view 3D Models from Single Images with a Convolutional Network (2016)</b> [[Paper]](https://arxiv.org/pdf/1511.06702.pdf) [[Code]](https://github.com/lmb-freiburg/mv3d)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/3d7ca5ad34f23a5fab16e73e287d1a059dc7ef9a/4-Figure2-1.png" /></p>

:camera: <b>View Synthesis by Appearance Flow (2016)</b> [[Paper]](https://people.eecs.berkeley.edu/~tinghuiz/papers/eccv16_appflow.pdf) [[Code]](https://github.com/tinghuiz/appearance-flow)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/12280506dc8b5c3ca2db29fc3be694d9a8bef48c/6-Figure2-1.png" /></p>

:space_invader: <b>Voxlets: Structured Prediction of Unobserved Voxels From a Single Depth Image (2016)</b> [[Paper]](http://visual.cs.ucl.ac.uk/pubs/depthPrediction/http://visual.cs.ucl.ac.uk/pubs/depthPrediction/)
<p align="center"><img width="30%" src="https://i.ytimg.com/vi/1wy4y2GWD5o/maxresdefault.jpg" /></p>

:space_invader: <b>3D-R2N2: 3D Recurrent Reconstruction Neural Network (2016)</b> [[Paper]](http://cvgl.stanford.edu/3d-r2n2/)
<p align="center"><img width="50%" src="http://3d-r2n2.stanford.edu/imgs/overview.png" /></p>

:space_invader: <b>Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision (2016)</b> [[Paper]](https://eng.ucmerced.edu/people/jyang44/papers/nips16_ptn.pdf)
<p align="center"><img width="70%" src="https://sites.google.com/site/skywalkeryxc/_/rsrc/1481104596238/perspective_transformer_nets/network_arch.png" /></p>

:space_invader: <b>TL-Embedding Network: Learning a Predictable and Generative Vector Representation for Objects (2016)</b> [[Paper]](https://arxiv.org/pdf/1603.08637.pdf)
<p align="center"><img width="50%" src="https://rohitgirdhar.github.io/GenerativePredictableVoxels/assets/webteaser.jpg" /></p>

:space_invader: <b>3D GAN: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (2016)</b> [[Paper]](https://arxiv.org/pdf/1610.07584.pdf)
<p align="center"><img width="50%" src="http://3dgan.csail.mit.edu/images/model.jpg" /></p>

:space_invader: <b>3D Shape Induction from 2D Views of Multiple Objects (2016)</b> [[Paper]](https://arxiv.org/pdf/1612.05872.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/e78572eeef8b967dec420013c65a6684487c13b2/2-Figure2-1.png" /></p>

:camera: <b>Unsupervised Learning of 3D Structure from Images (2016)</b> [[Paper]](https://arxiv.org/pdf/1607.00662.pdf)
<p align="center"><img width="50%" src="https://adriancolyer.files.wordpress.com/2016/12/unsupervised-3d-fig-10.jpeg?w=600" /></p>

:space_invader: <b>Generative and Discriminative Voxel Modeling with Convolutional Neural Networks (2016)</b> [[Paper]](https://arxiv.org/pdf/1608.04236.pdf) [[Code]](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
<p align="center"><img width="50%" src="http://davidstutz.de/wordpress/wp-content/uploads/2017/02/brock_vae.png" /></p>

:camera: <b>Multi-view Supervision for Single-view Reconstruction via Differentiable Ray Consistency (2017)</b> [[Paper]](https://shubhtuls.github.io/drc/)
<p align="center"><img width="50%" src="https://shubhtuls.github.io/drc/resources/images/teaserChair.png" /></p>

:camera: <b>Synthesizing 3D Shapes via Modeling Multi-View Depth Maps and Silhouettes with Deep Generative Networks (2017)</b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Soltani_Synthesizing_3D_Shapes_CVPR_2017_paper.pdf)  [[Code]](https://github.com/Amir-Arsalan/Synthesize3DviaDepthOrSil)
<p align="center"><img width="50%" src="https://jiajunwu.com/images/spotlight_3dvae.jpg" /></p>

:space_invader: <b>Octree Generating Networks: Efficient Convolutional Architectures for High-resolution 3D Outputs (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.09438.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/6c2a292bb018a8742cbb0bbc5e23dd0a454ffe3a/2-Figure2-1.png" /></p>

:space_invader: <b>Hierarchical Surface Prediction for 3D Object Reconstruction (2017)</b> [[Paper]](https://arxiv.org/pdf/1704.00710.pdf)
<p align="center"><img width="50%" src="http://bair.berkeley.edu/blog/assets/hsp/image_2.png" /></p>

:space_invader: <b>OctNetFusion: Learning Depth Fusion from Data (2017)</b> [[Paper]](https://arxiv.org/pdf/1704.01047.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/OctNetFusion-%20Learning%20Depth%20Fusion%20from%20Data.jpeg" /></p>

:game_die: <b>A Point Set Generation Network for 3D Object Reconstruction from a Single Image (2017)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf)
<p align="center"><img width="50%" src="http://gting.me/2017/07/17/pr-point-set-generation-from-single-image/ps3d_introduction.PNG" /></p>

:game_die: <b>Shape Generation using Spatially Partitioned Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.06267.pdf)
<p align="center"><img width="50%" src="http://mgadelha.me/sppc/fig/abstract.png" /></p>

:game_die: <b>DeformNet: Free-Form Deformation Network for 3D Shape Reconstruction from a Single Image (2017)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf)
<p align="center"><img width="50%" src="https://chrischoy.github.io/images/publication/deformnet/model.png" /></p>

:camera: <b>Transformation-Grounded Image Generation Network for Novel 3D View Synthesis (2017)</b> [[Paper]](http://www.cs.unc.edu/~eunbyung/tvsn/)
<p align="center"><img width="50%" src="https://eng.ucmerced.edu/people/jyang44/pics/view_synthesis.gif" /></p>

:camera: <b>Tag Disentangled Generative Adversarial Networks for Object Image Re-rendering (2017)</b> [[Paper]](http://static.ijcai.org/proceedings-2017/0404.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Tag%20Disentangled%20Generative%20Adversarial%20Networks%20for%20Object%20Image%20Re-rendering.jpeg" /></p>

:camera: <b>3D Shape Reconstruction from Sketches via Multi-view Convolutional Networks (2017)</b> [[Paper]](http://www.cs.unc.edu/~eunbyung/tvsn/)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/SketchModeling/SketchModeling_teaser.png" /></p>

:space_invader: <b>Interactive 3D Modeling with a Generative Adversarial Network (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.05170.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DCsPKLqXoAEBd-V.jpg" /></p>

:camera::space_invader: <b>Weakly supervised 3D Reconstruction with Adversarial Constraint (2017)</b> [[Paper]](https://arxiv.org/pdf/1705.10904.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Weakly%20supervised%203D%20Reconstruction%20with%20Adversarial%20Constraint%20(2017).jpeg" /></p>

:gem: <b>Exploring Generative 3D Shapes Using Autoencoder Networks (Autodesk 2017)</b> [[Paper]](https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Exploring%20Generative%203D%20Shapes%20Using%20Autoencoder%20Networks.jpeg" /></p>

:pill: <b>GRASS: Generative Recursive Autoencoders for Shape Structures (SIGGRAPH 2017)</b> [[Paper]](http://kevinkaixu.net/projects/grass.html)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/grass/teaser.jpg" /></p>

## Style Transfer
<b>Style-Content Separation by Anisotropic Part Scales (2010)</b> [[Paper]](https://www.cs.sfu.ca/~haoz/pubs/xu_siga10_style.pdf)
<p align="center"><img width="50%" src="https://sites.google.com/site/kevinkaixu/_/rsrc/1472852123106/publications/style_b.jpg?height=145&width=400" /></p>

<b>Design Preserving Garment Transfer (2012)</b> [[Paper]](https://hal.inria.fr/hal-00695903/file/GarmentTransfer.pdf)
<p align="center"><img width="30%" src="https://hal.inria.fr/hal-00695903v2/file/02_WomanToAll.jpg" /></p>

<b>Analogy-Driven 3D Style Transfer (2014)</b> [[Paper]](http://www.chongyangma.com/publications/st/index.html)
<p align="center"><img width="50%" src="http://www.cs.ubc.ca/~chyma/publications/st/2014_st_teaser.png" /></p>

<b>Elements of Style: Learning Perceptual Shape Style Similarity (2015)</b> [[Paper]](http://people.cs.umass.edu/~zlun/papers/StyleSimilarity/StyleSimilarity.pdf)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/StyleSimilarity/StyleSimilarity_teaser.jpg" /></p>

<b>Functionality Preserving Shape Style Transfer (2016)</b> [[Paper]](http://people.cs.umass.edu/~zlun/papers/StyleTransfer/StyleTransfer.pdf)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/StyleTransfer/StyleTransfer_teaser.jpg" /></p>

<b>Unsupervised Texture Transfer from Images to Model Collections (2016)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/siga16_texture_transfer_small.pdf)
<p align="center"><img width="50%" src="http://geometry.cs.ucl.ac.uk/projects/2016/texture_transfer/paper_docs/teaser.png" /></p>

<b>Learning Detail Transfer based on Geometric Features (2017)</b> [[Paper]](http://surfacedetails.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://surfacedetails.cs.princeton.edu/images/teaser.png" /></p>
