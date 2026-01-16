## Defending Digital Integrity: Advances in Media Forgery Analysis Research and Cybersecurity Development

---
### Abstract

In light of the widespread availability of instruments for gathering and processing multimedia signals, a growing body of data suggests that images and videos should not be relied upon as reliable evidence. Since this is a possibility, digital broadcasters need to know how to tell if the multimedia content they've received is authentic or not. Experts in signal processing have been developing efficient solutions for media forgery analysis to address these problems. These methods aim to reconstruct the processing history of the multimedia data at issue and validate its sources. These methods rely on the idea that most changes are irreversible and leave "footprints" in the reconstructed signal that can be used to reduce the processing steps that came before. To highlight the list of challenges, tools, datasets, and techniques for forgery detection. This study provides a summary of multimedia forgery based on the studies presented in the literature.
**Keywords** Media forgery detection Image manipulation Noise artefacts Compression artefacts Intra-Frame forgery detection

---
### Introduction

We live in an era where multimedia technology, such as contemporary recorders and cell phones, is becoming increasingly popular. These recordings are constantly edited for various purposes with the development of more publicly available picture and video editing technologies [1].

The internet has undergone significant changes, with the unplanned production of accessible and dependable digital media, and the emergence of altering tools for instance, the widespread adoption of low-cost, affordable video-editing software such as Lightworks, Photoshop, Adobe Premiere, and Celera-resulting in various forgeries and unauthenticated expressions of digital data. The significance of forensics has been shown [2, 3].

Fake multimedia has become a significant issue in recent years, particularly following the introduction of so-called deepfakes, which are photos and videos that have been manipulated using advanced deep-learning algorithms. Creating realistic modified media assets using this technology may be quite simple, assuming one has access to enormous volumes of data. With advances in deep learning [4, 5], manipulations might appear incredibly realistic and difficult to detect [6]. It is critical to consider the intent underlying such manipulations. For example, the media might be managed to spread misinformation or perpetrate financial fraud [7, 8], false porn movies could be created to blackmail people, or fake news campaigns could be built to sway public opinion. It can also decrease faith in journalism involving serious and credible sources in the long run. Figure 1 depicts some widely circulated deepfakes on the Internet. Attempts to influence social discourse, elections, and the behavior of individuals in a civilized society [9] have fueled experiments in multimedia forensics [10, 11].

Image forensics [12-14] and video forensics [5, 15] have review papers. This paper surveys the entire landscape of multimedia forgery detection, making three specific contributions.

* **Comprehensive Taxonomy of Detection Approaches.** A taxonomy organizes the techniques into image-based methods and video-based methods. Within each branch, the authors further categorize the approaches by the artifacts they exploit noise signatures, compression fingerprints, color filter array (CFA) patterns, and explicit manipulations and by the attack vectors they employ, whether intra-frame or inter-frame.
* **Tools and Software for Practitioners.** The paper then creates a side-by-side inventory of both commercial suites and open-source utilities that analysts can reach for today. The inventory lists feature sets, common deployment scenarios, licensing hurdles, and a few user testimonials to ground the comparison in real fieldwork.
* **Survey of Benchmark Datasets.** Researchers often lean on benchmark datasets, so the authors compile the most widely referenced collections for still images and moving footage. Each entry is assessed for resolution, manipulation taxon, known weaknesses, and how well it reflects the kind of media one might encounter in a court of law.
* **Emerging Threats and Ongoing Obstacles.** Deepfakes and other next-generation forgeries present headaches that older methods cannot unravel easily, and the text outlines those headaches squarely. Compression smears, real-time demand, tiny training corpora, and the muddy waters of legal and ethical accountability are cited as the sticking points moving forward.

Identification of Research Gaps and Future Directions: A close examination of the existing literature reveals persistent weaknesses in media-forensic toolkits, notably the shortage of deep-learning architecture that remain practical and reliable when confronted with noisy, low-resolution material or deliberate adversarial attacks. The observations gathered here are intended to provide both academic investigators and industry practitioners with a clear baseline, encouraging new experiments that push the boundaries of multimedia forgery detection.

---
### Methodology

The present survey aims to map, in an orderly manner, the research terrain of multimedia forgery detection, with a particular emphasis on the manipulation of image and video content. A multi-tiered strategy, including selection, classification, analysis, and synthesis, was employed to maintain a disciplined and transparent review process. Each stage, given in Fig. 2, of that strategy, is detailed in what follows so that the work may be reproduced and its choices understood.

#### Literature Search Strategy

Identifying solid, relevant studies proved to be the necessary first step. That step began with a methodical hunt across several well-established databases: IEEE Xplore, SpringerLink, Elsevier ScienceDirect, the ACM Digital Library, Wiley Online Library, MDPI, and, for fresh papers and preprints, a targeted check of Google Scholar. Careful combinations of search terms and Boolean connectors, paired with date and document-type limits, shaped the results. Key phrases such as *image forgery detection*, *media forensics*, *copy-move forgery*, *compression artifacts*, and *deepfake detection* featured prominently, as did narrower entries like *intra-frame manipulation* and the study of *anti-forensic tactics*. Dates were capped in 2024, and language was restricted to English text. This allowed the query to capture the bulk of relevant activity since 2005 while excluding older references that no longer met contemporary standards.

#### Inclusion and Exclusion Criteria

A systematic approach required that only the most pertinent and methodologically sound papers be included in the review pool. The following benchmarks ultimately dictated inclusion.

* Peer-reviewed journals and conference proceedings were the baseline requirement. Only studies published through established review channels could be considered authoritative.
* Original contributions mattered. Papers that proposed fresh algorithms, benchmarking datasets, or end-to-end forensic tools for detecting image and video fraud all qualified.
* Machine learning or deep learning frameworks were a must. Research rooted entirely in classical methods, although informative, did not align with the contemporary focus.
* Surveys and meta-analyses were not excluded outright. Such overviews often lay the groundwork and help consolidate scattered findings.
* In contrast, non-peer-reviewed blogs and white papers were set aside. Despite occasional insights, they lacked the editorial rigor essential for academic discourse.
* Works that concentrated solely on sociological impact without delving into code or computation were filtered out as tangential.
* Duplicate publications, or those updated within a single calendar year by the same authors, were pruned to avoid redundancy.
* Papers that ventured into cybercrime beyond multimedia, such as network packet inspection or malware tracing, were deemed outside the scope of the bibliography.

#### Data Collection and Extraction

Once the final set of articles was clear, each document underwent a granular examination. Key facets were lifted and logged in a standardized framework.

* First, the paper's objective, including the fraud modality addressed, still image, moving frame, or a blend, was noted.
* Core methodologies surfaced next. Algorithms, pipelines, and any bespoke software mentioned by the authors were cataloged alongside the text.
* Datasets received equal attention. Publicly available collections were distinguished from private or self-constructed samples to gauge portability.
* Evaluation criteria are followed closely. Precision, recall, AUC, and other numerical benchmarks were recorded in their original units for later meta-analysis.
* A strengths-and-limitations commentary rounded out each line entry, capturing novelty claims as well as acknowledged weaknesses.
* Publication metadata-finally-included year, venue, and DOI or conference URL to facilitate quick retrieval after the synthesis phase.

All extracted details converged in a master review matrix, enabling side-by-side comparison across methodological, empirical, and contextual dimensions.

#### Classification and Organization of Content

For the sake of transparency and analytical depth, each topic was grouped into distinct thematic buckets. The first, **Forgery Detection Techniques**, covers noise artifacts, CFA inconsistencies, manipulation fingerprints, and the now-familiar signs of deepfake tampering. A second cluster, **Video Forgery Types**, distinguishes between intra-frame edits, inter-frame cuts, compression tricks, and the lingering cat-and-mouse game of anti-forensic countermeasures. A third section, **Tools and Software**, assesses both commercial suites and open-source codebases, categorizing them by whether they enhance images, authenticate files, or flag manipulations outright. **Benchmark Datasets** follow next, arrayed by medium (still images or moving clips), by the tampering the footage undergoes, by resolution, and sometimes by the credibility of the origin. **Challenges and Research Gaps** then confront the hard truths of lossy compression, the patchy availability of testbeds, the uneven robustness of algorithms, and the ever-looming need for real-time processing speed. **Future research directions** conclude the paper with a list drawn directly from those gaps and buzzworthy trends emerging in recent studies.

#### Synthesis and Critical Analysis

A side-by-side comparison of findings began to clarify which methods truly stand up in the messy ecosystem of social media footage and low-bandwidth CCTV. Accuracy, generalizability, dataset hunger, and computational cost all entered the scoring sheet. One clear trend is the growing domination of deep-learning approaches; almost every paper now touts some neural architecture at its core. Another pattern is the rising demand for test sets that mimic the unusual characteristics of everyday video, including both amateur deepfakes and authentic recordings that have been edited and recorded multiple times. The review began, rather tediously, with a side-by-side appraisal of papers that ask the same question but follow different trails. Researchers marked what worked, what stumbled, and tried, not always easily, to chart where the discipline has wandered over the last ten years.

#### Scope and Limitations

For sheer practicality, the write-up focuses on **passive forgery detection**; investigators rarely have access to raw, lossless material, so that choice seemed obvious. Space allows for a broad selection of algorithms and datasets, yet discussions of legal, ethical, or policy implications are noticeably left in the wings. Because deepfake research still advances rapidly, a handful of preprints and lab notes released in the last few weeks may well slip through the net and remain unmentioned.

---
### Background of Image Forgery

Images in multimedia are generated using an acquisition system that consists primarily of hardware and software. Then, a series of internal processing procedures, such as color correction, enhancement, and compression, are performed. These components differ in their implementation and parametrization depending on the camera model and provide crucial hints for forgery analysis and multimedia analysis. In addition to introducing artifacts for forgery analysis and identification, alterations made by a malicious user may also introduce such artifacts [16]. In reality, the image formation process within a camera involves several hardware and software processes that are unique to each camera, leaving various fingerprints on the recorded image. Similarly, the out-of-camera editing process may generate unique traces and disrupt fingerprint-like camera-specific patterns, allowing the attack to detect them reliably (Fig. 3). Most of these traces are quite subtle and cannot be detected visually [17]. Specifically, multimedia forgery analysis tools search for various artifacts caused by in-camera or out-of-camera processing.

Today, the proliferation of the Internet (particularly search engines and social networks) and the development of media editing software (such as Adobe Photoshop and others) enable the creation, editing, and sharing of massive amounts of images, audio, and videos daily. They result in billions of views of this information rendered to a diverse and large audience. Furthermore, this information is rendered to various platforms, including mobile devices, desktop computers, and televisions. Therefore, verification of the medium is required in many different circumstances.

The recent and rapid advancement of artificial intelligence (AI) has led to the development of several new technologies, including deepfakes, computer-generated imagery (CGI), and anti-forgery analysis techniques. All of these pose a significant threat to the trustworthiness of content produced by the media (images/videos/audio/documents/ text). Researchers are working to develop technologies that can detect manipulated media, discover inconsistencies, retrieve the provenance of digital content from disinformation, and ensure that digital content can be trusted and authenticated. Multimedia forgery analysis aims to equip investigators with the computational tools necessary to determine the legitimacy of digital media recording and its point of origin. For such tools to be developed, it is necessary to have competence in computer vision and machine learning, and for the relevant communities to dedicate their attention and energy to working on challenges of this nature.

Image manipulation is not a brand-new issue. Since the invention of photography, people have been manipulating images for various powerful reasons. Tools for modifying images and videos, such as Photoshop, have been around for quite some time. Images can be easily manipulated using traditional signal processing methods, producing real results that can fool even the most vigilant observer. Research in multi-media forgery analysis has been ongoing for at least 15 years [18, 19]. It is gaining increasing attention from the academic community, as well as significant information technology (IT) firms and funding agencies. Detection of image manipulation has become a crucial challenge in multimedia forgery analysis. As a result, numerous forgery analysis methodologies have been developed over the past decade to assess the authenticity and processing history of digital images. It has been noted that image manipulations generally leave traces specific to the editing an image has experienced [20]. Therefore, researchers develop forgery analysis algorithms that identify attributes associated with these traces and utilize them to detect image manipulations that have been specifically targeted.

---
### Forgery Tools Overview

With the advancement of technology, tools to assist investigators in analyzing the vast amount of evidence on multimedia devices are becoming increasingly important. To present credible evidence in court, investigators must ensure the accuracy of their instruments and their effective use [21]. Over the last decade, several tools have been developed to assist multimedia forensic investigations.

* **Teel Tech Canada** offers a range of technologies to support video forensics investigations. The **Corepro** tool is utilized for picture comparison via reverse projection, which provides information about an intriguing entity. For video enhancement, **impress** programs employ several filters. The **Mandet** tool helps investigators determine the validity of a video. After registering, all three tools provided by Teel Tech Canada are available for free use.
* **Cognitech** offers two video forensic investigation tools. The **Video Investigator** tool is intended to enhance videos for research purposes. An **Auto-measure** tool is a piece of photogrammetry software that is used to measure biometrics and scenes. Both of these tools are paid for and do not provide free trials.
* **Amped Software** offers two video forensic analysis software packages. **Amped Five** is video-enhancing software, while **Amped Authenticate** is a solution for detecting counterfeit or manipulation content. Both Amped tools are available for purchase only.
* **Forensics Video-FA** software from **DiViLine Expert Systems** is used for motion detection and information extraction from video files. It is not free; however, a trial version is available.
* Another software solution for video enhancement and tamper detection is **Video Cleaner**. It is freely downloadable and simple to install.
* The **dTective** software solution from **Ocean Systems** can be used to enhance video data for forensic investigations.
* **Kinesense** software is capable of object recognition and augmentation. It is paid software; however, a free trial is available.
* **Vocord's Video Expert** provides video improvement, authentication, and facial recognition services, as well as generates investigative reports.

Finally, forensic investigators have access to a large choice of forensic video products. The investigators can determine their demands, select a product that best meets their individual needs, or choose an existing tool to help them with their investigations. Table 1 summarizes the key video forensics products available for inquiry.

| Company | Tool | Functions |
| :--- | :--- | :--- |
| Teel Tech Canada | COREPRO | Image Comparison |
| | IMPRESS | Enhancement |
| | MANDET | Forgery Detection |
| | VIEWPOINT | Enhancement |
| Cognitech | Video Investigator | FaceFusion3D, Adaptive Blur, FrameFusion |
| | AutoMeasure | Photogrammetry |
| Amped | FIVE | Enhancement |
| | Amped Authenticate | Tamper Detection |
| DiViLine Expert Systems | Forensic Video-FA | Information Retrieval, Motion Detection |
| Doug Carner | VideoCleaner | Enhancement, Tamper Detection |
| Ocean Systems | dTective | Enhancement |
| Kinesense | Kinesense | Retrieval/Object Recognition/ Enhancement |
| Forevid | Forensic Video Analyzer Software | Record Screen, Bookmark Framed, Resize, Enhance, Deinterlace, Rotate. Meta Details |
| VOCORD | Video Expert | Enhancements, Authentication, Forensic Reports |

---
### An Overview of some Forgery Detection Techniques

This section discusses the techniques used in image and video forensics. We will first go through image and video counterfeit detection techniques, followed by information extraction techniques.

#### Forgery Detection Techniques for Image

**a. Noise artifacts.**
Noise is the inherent fingerprint of digital media and is frequently used to detect forgeries. Exposing noise artifacts is one of the most effective approaches presented in many forgery analysis scenarios. Most noise-based techniques presume that the noise is uniform throughout the entire image and can be represented by white Gaussian noise [22]. Therefore, the noise artifacts stay consistent throughout digital media if its contents are not manipulated [22].

Based on this fact, a robust descriptor, **FFTDRLBP** (Fast Fourier Transformation - Discriminative Robust Local Binary Patterns), is presented in [11]. This descriptor first estimates noise patterns using FFT and then encodes differences in noise patterns using DRLBP. The retrieved features are sent to a Support Vector Machine (SVM) to determine whether the image is genuine or manipulated. **Noise Level inconsistencies** are also exploited by [23-25].

Noise pattern is also the foundation for the so-called **Error Level Analysis (ELA)** [24, 26-28]. In most methods used to detect manipulation [26-28], the images are subjected to Error Level Analysis, which suppresses the image's primary features and amplifies its manipulation-related latent features, before being fed into the deep learning model. Most of the time, the noise level is insufficient to indicate the existence of manipulation and may lead to incorrect guesses in this field. Therefore, in [29], **high-pass noise residual** is adopted in the new research to extract the rich features that can be collected and then fed to a complex classifier to make a final decision. For example, A model for detecting image splicing tampering based on dual-channel dilated convolution is performed in [30]. It combines deep and superficial image characteristics to locate the tampered area precisely. In this model, the first channel captures noise characteristics using a set of high-pass filters to identify inconsistencies in noise between the true and tampered regions. Then, in conjunction with the attention mechanism, the second channel collects RGB image features by dilated convolution and locates the altered region. Then, the bilinear pooling layer combines the characteristics generated by the two channels for each region of interest. Finally, tamper classification and boundary box regression are implemented. Besides, an enhanced **Adaptive Spatial Rich Model (ASRM)** is presented in [31] to assist in mining subtle noise characteristics using learnable high-pass filters.

On the other hand, the camera's sensor can provide an abundance of valuable clues. Due to manufacturing defects, the sensor elements exhibit minor deviations from their predicted function. These deviations generate a stable, noise-like pattern known as **photo-response non-uniformity (PRNU) noise** [16]. All images captured by a particular camera have traces of its PRNU pattern, which can be viewed as a form of a fingerprint [16]. If a portion of the image is manipulated, the associated PRNU pattern is erased, allowing the modification to be detected [16]. The authors of [32] use forgery localization based on photo-response non-uniformity (PRNU) noise as an illustration and propose a segmentation-based forgery localization scheme that exploits the local homogeneity of visually imperceptible clues to mitigate the limitations of existing segmentation approaches that are based primarily on visually perceptible content. This noise model requires significant prior knowledge, as a specific number of source device images or the device itself must be provided. On the other hand, it is incredibly potent, as it can detect all types of attacks with equal accuracy. The primary issue is estimating a single image during testing, as the PRNU pattern is a weak signal that can be easily overpowered by improperly deleted visual content [33]. On the other hand, this approach can also be extended to blind scenarios, where no prior knowledge about the camera is available, as long as an appropriate clustering procedure detects images that share the same PRNU [34, 35].

**b. Compression artifacts.**
Utilizing compression artifacts has been a mainstay of image forgery analysis for decades. The numerous approaches, particularly for JPEG-compressed images, can be categorized according to the cues upon which they rely. For example, utilizing the so-called **lock artifact grid** is a common color filter array (CFA) technique. In the case of splicing or copy-move manipulations, the BAGs (Block Artifact Grids) of the inserted object and the host image frequently mismatch, allowing for detection. Several BAG-based approaches have been proposed [36, 37], some as recently as [38].

Another effective strategy utilizes **double compression traces**. When a JPEG-compressed image is subjected to local manipulation and then recompressed, double compression artifacts appear elsewhere on the image, except in the manipulated region [38]. Diverse methods have been developed to detect this type of manipulation in digital images of the well-known JPEG format. For example, the authors of [39] suggest a **part-level middle-out learning technique** for detecting double compression using an architecturally efficient classifier. Initially, double-compressed data with varied JPEG coder settings is represented in feature space as a limited number of coherent clusters known as parts. Next, the behavior of a set of notable **Benford-based features** is depicted. The challenge of detecting double JPEG compression in the family of feature engineering-based techniques is then characterized as a part-level classification problem, aiming to cover all conceivable JPEG quality level combinations by utilizing the newly discovered information.

Another Detection model of double JPEG compression is presented in [40], which utilizes **component convergence during multiple JPEG compressions**. First, a comprehensive investigation of the pipeline in subsequent JPEG compressions reveals that rounding/truncation errors, as well as JPEG coefficients, tend to converge following several recompressions. Based on this information, the **backward quantization error (BQE)** is defined, and it is discovered that the ratio of non-zero BQE for single compression is more significant than that for double compression. In addition, a multi-threshold technique is developed to capture the statistics of the number of unique JPEG coefficients between two successive compressions, utilizing the convergence property of JPEG coefficients. To detect double JPEG compression, the statistical properties of the dual components are concatenated into a 15-D vector. Suppose the primary quality element is more significant than the secondary quality component in the double JPEG compression images. In that case, the authors of [41] present a novel feature extraction approach based on **optimum pixel difference (OPD)**, a new measure for blocking artifacts, to meet this challenging task. Initially, the three-color channels (RGB) of a decompressed JPEG color image are transferred into spherical coordinates to determine the amplitude and two angles (azimuth and zenith). Then, 16 histograms of OPD are calculated along the horizontal and vertical directions in amplitude and two angles, respectively. A set of features is created by organizing the bin values of these histograms, which is subsequently used for binary classification. This type of artifact varies depending on whether the two compressions are spatially aligned or not; however, suitable detection methods [42, 43] have been proposed for both cases.

Another method employs so-called **JPEG ghosts** [44], which appear in the manipulated region when two JPEG compressions with the same quality factor (QF) are used. The target image is compressed at all QFs and evaluated to highlight ghosts.

Exploiting compression artifacts for detecting video manipulation is also conceivable, although this is considerably more challenging due to the complexity of the video coding process [16]. Existing approaches only evaluate identifying videos with a predetermined Group of Pictures (GOP). The current direction in research is to investigate double compression detection methods for H.264 videos with both fixed and adaptive GOP structures. Considering that video may contain adaptive GOPs due to fast-moving content or scene changes [45], provides an example of how our double compression detection approach utilizes **temporal segmentation** to divide the video into static and rapid periods, containing normal fixed and adaptive GOPs, respectively. Then, new artifacts are assessed based on the sequence of **frame byte count (FBC)**. Combining the artifacts in the video's static and fast periods produces a feature sequence made of distinguishable distances. To uncover the inherent property of the feature sequence, a scoring technique is created to decide whether double compression is appropriate.

Videos from Instagram, WeChat, TikTok, and other platforms are frequently compressed and shared on social media. Consequently, the ability to recognize compressed Deepfake videos becomes a critical concern. In [46], a **two-stream technique** is proposed that analyzes the frame-level and temporal level of compressed Deepfake videos. The suggested frame-level stream gradually prunes the network to prevent the model from fitting the compression noise. Video compression adds a great deal of redundant information to each frame. A temporality-level stream is employed to extract temporal correlation characteristics, addressing the issue that the temporal consistency of Deepfake movies may need to be considered. The suggested method outperforms state-of-the-art techniques for detecting compressed Deepfake videos when paired with scores from two streams.

**c. A color filter array artifact.**
A **color filter array (CFA)** is used by most image acquisition devices to acquire color images. A color filter array is designed to gather color data from a single sensor, sampling it for each pixel. Each pixel stores the absorbed light value of a single color. For example, the **Bayer pattern** [10], shown in (Fig. 4a), is used in most commercial cameras. To construct a complete color image from the mosaic image obtained from the CFA, it is necessary to restore the missing pixels for each color channel. This process is known as **color interpolation or color demosaicing** [10]. It is a crucial component of image signal processing.

Since the appearance of commercial cameras, various algorithms for color demosaicing have been developed. Interpolation-based techniques such as bilateral and bicubic interpolation appeared relatively early in image processing and computer vision. However, interpolation-based techniques produce undesirable effects such as the zipper effect, false color, and blurring. These effects produce periodic patterns, depending on the interpolation techniques used. These periodic patterns are disrupted each time a manipulation happens. In addition, because CFA configuration and interpolation algorithms are model-specific [47, 48], when a region is spliced into an image taken with a different camera model, its periodic pattern will appear irregular [11, 22], exposing multi-media forgeries in terms of using CFA configuration with an efficient noise fingerprint included for splicing forgery localization [11]. After interpolation, it is predicted that the noise of the interpolated pixels will be reduced. Therefore, the only factor influencing the noise levels of nearby acquired and interpolated pixels is the interpolation procedure, which is constant in the original image. A **dual-tree wavelet-based denoising approach** is used to extract the noise from the green channel and determine the standard deviation of the noise for both acquired and interpolated pixels. The geometric mean of the noise standard deviations is then used to calculate the noise level of the captured and interpolated pixels. Finally, a fingerprint to identify tampered areas can be created using the **noise level ratio between acquired and interpolated pixels**.

In [10], the analysis is expanded to include a **covariance matrix**, which is used to rebuild the R, G, and B channels of an image. The inconsistencies of the CFA interpolation pattern are used to extract forgery analysis features. Then, these forgery analysis features were used for coarse-grained detection, while texture strength features were used for fine-grained detection. A method of edge smoothing was then used to achieve exact localization.

The authors in [47] revealed CFA artifacts across different domains through **higher-order statistical analysis** based on the **Markov transition probability matrix (MTPM)**. First, the image is re-interpolated using the four most prevalent Bayer CFA patterns. For simplicity, the re-interpolation method employs a bilinear interpolation scheme. To examine CFA discrepancies, the difference between the provided image and its re-interpolated versions is then calculated. Finally, the target difference image is picked corresponding to the maximum sum, which is then processed to evaluate the MTPM-based second-order statistical feature.

Combining multiple multimedia forgery analysis tools to detect media manipulation is possible. For example, the authors of [48] Incorporated **noise fingerprint with a CFA setup** for splicing forgery localization. It is anticipated that the noise of interpolated pixels will be reduced after interpolation, and the relationship between the noise levels of adjacent acquired and interpolated pixels is only connected to the interpolation procedure, which is constant in the original image. A dual-tree wavelet-based denoising technique is employed to extract noise from the green channel and compute the standard deviation of noise for both acquired and interpolated pixels. The noise level of acquired and interpolated pixels is then calculated using the geometric mean of the standard deviations of the noise. The ratio of noise levels between collected and interpolated pixels can be used as a fingerprint to identify altered locations.

Another example is found in [22], where two algorithms are combined. The first approach is an **Error Level Analysis (ELA)** algorithm that can be used as an initial filter to detect the existence of splicing in an image. It emphasizes pixels with a distinct compression level. The second algorithm is a **digital image authentication method** based on the estimated quadratic mean error of the CFA interpolation pattern. Using chromatic interpolation methods, it identifies forged color images.

**d. Manipulation artifacts.**
In addition to artifacts associated with re-compression, the manipulation process frequently leaves behind valuable traces. Indeed, when a new object is inserted into an image, it often requires several postprocessing procedures to be properly integrated into its new environment. These include geometric transformations such as rotation and scaling, contrast adjustment, and blurring to soften the object-background borders.

**Copy-move forgery** is a prevalent type of digital image manipulation. In copy-move forgeries, a portion of an image is duplicated within the same image but in a different location. For the restoration of image credibility, it is necessary to build an effective and robust technique for detecting such forgeries [16, 18]. Detection methods for this type of manipulation are generally categorized into **Block-based and Keypoint-based approaches** [49]. As a result, numerous articles concentrate on detecting these fundamental actions as proxies for suspected forgeries.

In a **Block-based technique** (Fig. 5a), the image is divided into small, overlapping, or non-overlapping blocks. These blocks are usually nearly square or rectangular. The extracted features are compared to each other to determine which blocks or characteristics match. In contrast to Gaussian noise and JPEG compression, block-based forgery techniques are effective. Among the disadvantages of Block-based approaches is the need for more clarity in determining the proper block size. Additionally, small blocks increase the computational cost of matching and lack robust characteristics. Furthermore, large blocks cannot be utilized to detect small regions of forgery and tend to detect identical portions as duplicates [50]. For example, the authors of [51] propose a new **Tetrolet transform-based method** for detecting copy-move image forgery. In this technique, the input image is first divided into overlapping blocks, and then the Tetrolet transform is used to extract four low-pass coefficients and twelve high-pass coefficients from each block. Then, feature vectors are arranged lexicographically, and comparable blocks are detected by comparing retrieved Tetrolet features. Even when the cloned parts have been subjected to blurring, color reduction, changes in brightness and contrast, rotation, scaling, and JPEG compression, the suggested technique can recognize and locate the duplicated regions in the images with great accuracy.

In the **key point-based method** (Fig. 5b), feature vectors are calculated for regions with high entropy without subdividing the image. The feature vectors are then examined for matches. Techniques based on key points help spot forgeries that have been scaled or rotated. However, their primary limitations include the need to match multiple key points and the requirement for filtering approaches, such as **Random Sample Consensus (RANSAC)**, to reduce false positives [49]. An example of this approach is found in [52], which suggests that a robust and improved algorithm for detecting copy-move forgery has been created by combining **block-based DCT** (Discrete Cosine Transform) and **keypoint-based SURF** (Speeded-Up Robust Features) techniques on the MATLAB platform.

A third strategy based on **image segmentation** (Fig. 5c) was recently proposed [1]. The primary disadvantage of this method is that segmentation cannot distinguish between foreground and background regions (see Fig. 5.c). To overcome the limitations of Block-based, Keypoint-based, and Segment-based approaches, a new method exploits **image blobs and Binary Robust Invariant Scalable Keypoints (BRISK) features** [49]. First, it locates regions of interest, called image blobs, and BRISK features in the analyzed image, and then identifies BRISK key points within the same blob. Finally, perform a matching process between BRISK key points in different blobs to find similar key points for copy-move regions.

---
#### Forgery Detection Techniques for Video

**A. Compression Video Techniques.**
The video has two spatial dimensions and one temporal dimension. Predictive coding is used to eliminate temporal redundancy, whereas transformation field coding is used to reduce geographical duplication. In video compression, there are three types of frames: **P-frames** (forward-predicted), **I-frames** (intra-coded), and **B-frames** (bidirectionally predicted). I-frames rely on spatial redundancy and are encoded without a reference frame; P frames forecast using I or P frames; and B frames forecast using past and future frames. As a result, frames (P and B) exploit both time and spatial redundancy whenever possible. As a result, B frames compress data better than P frames. **GoPs (Group of Pictures)** are used to categorize video sequences. The I-frame is the initial frame in the GOP group, surrounded by frames B and P. Traditional video encoders, like the ones called MPEG-1 and MPEG-2, have a certain GOP form, with each GOP ensemble holding a defined number of frames [53]. This has an impact on both video quality and the compression ratio. The H.264 video codecs [54] provide a dependable and extensible GOP structure. Frames in **Adaptive GOP (AGOP)** can be extended up to 250 frames, depending on the video content. The number of frames in a series of photos in a dynamic movie will be smaller than the background of the frame, and the contents will change swiftly. This improves both compression and visual quality. Compression techniques are used to compress movies due to storage and distribution constraints. In [55-61] examined how to identify movies as altered if they were reduced twice or more. Wang and Fred [57, 61] employed findings from the **statistical analysis of P-frame error estimates** and the **distribution of the coefficients of discrete cosine transform (DCT) for massive blocks (MB) of I-frames** in their works.

**B. Anti-forensic video techniques.**
Anti-forensic video techniques were developed to deceive forensic investigators by removing or concealing evidence that has been tampered with or altered. While forensic techniques can help detect digital video tampering, the majority of them will be ineffective if the forger employs an anti-forensic approach. Anti-forensic techniques are based on the premise that removing or reducing traces produced after a video has been manipulated results in additional data that must be analyzed further to detect the forgery [62].

Both Stamm and Liu [57] made a case for the use of video to fool one of the forensic methods described in [63], notably those that rely on **GOP desynchronization**. The authors [64] believe that the simplest way to make forging indistinguishable is to increase the accuracy of prediction for all images to levels that appear as bulges. At the same time, the high points in error are undetectable due to asynchrony. They alter the encoder to assign a set number of force variables to zero, even though they are not null, to increase prediction errors [62]. The quality of the video will not change because the error will be saved throughout encoding and given before reproduction; furthermore, the writer(s) decided to reduce the dimension to zero initially with the lowest so that the actual error propagates through a large number of sectors and the resulting alteration is very hard to detect. According to the authors, Wang et al. used a different detection method. The same study [63] can be combated by employing counter-forensic approaches developed for still images, notably those that conceal JPEG quantization effects [64]. A real counter-direct strategy also applies to images for camera-based approaches: all that is required is an increase in the resultant video (even by a modest factor) and then editing it [63].

**C. Temporal (Inter-Frame) Tampering forgery detection.**
**Inter-frame** refers to interfering with the sequence of frames in the video, for instance, by **copying, adding, or removing frames** [65]. This is a counterfeit because it can duplicate frames or alter the order in which the video frames appear. There are three types of inter-frame video forgery, as shown in Fig. 6:
* **Inserting frames:** Adding a set of frames in the same order as the original frames.
* **Frame deletion:** Frame deletion is the process of removing a few frames from a video's sequence of frames. It is known that the fake video will be smaller than the real one.
* **Duplicating a frame** entail pasting and copying the same object. Frame duplication in this context refers to copying and pasting a group of frames from one video into another.

In other words, altering the content of a single video by copying, moving, or splicing it constitutes an inter-frame video forgery [66]. To detect copying and video counterfeiting, various strategies have been developed and implemented over time [67]. The frames were assembled in a separate location in a video in Wang and Farid's 2007 study [67]. The frames were added in the same order and replicated using any location in the video. Other techniques, including **residue characteristics** and **transformation in cross-modal subspace**, were suggested in 2010 [68].

Some researchers employed the **peak-signal-to-noise ratio (PSNR)** to calculate the amount of motion that occurs in a video [69]. This was also used to gauge content that had been altered. According to [68], **3-D ballistic motion** may be detected in flights within the movie, indicating that gravity is influencing the object's route in this manner. Similar to how [70] used the **motion error prediction technique** to identify the addition or removal of frames in the video [71], applied **geometric video approaches** to locate the objects that moved in a video sequence. For example [72], advanced the theory that when a video undergoes double MPEG compression, it employs static and temporal artifacts.

The **prediction of footprint variation (PFV) pattern** was used by the researchers of [69] to identify anomalous P-frames as outliers. They further improved these findings by applying various kinds of **motion vectors (VMVs)** to identify the location of the forgery. The authors of [67] employed a **histogram of oriented gradients (HOG)** as a characteristic for interframe forgery detection. **Grabb's test** was used to identify irregular points based on correlation coefficients. Additionally, a **motion energy image (MEI)** was used to find duplicate and shuffled frames. To detect frame duplication [70], employed the time-based average of each shot rather than every frame. To detect frame duplication, **grey-level combination matrix (GLCM)** information was derived for feature vectors and compared to neighboring vectors. To identify fabricated frames [62], employed a thorough **convolutional neural network-based method** that leveraged the temporal and spatial correlations of frame data.

**D. Intra-frame forgery detection.**
For video counterfeiting, two sorts of **intra-frame** actions can be performed: **copy and paste operations and crop operations** [66]. Some elements from outside the frame are introduced during the copy-and-paste procedure to create a new fake video sequence. The crop operation, which hides certain specific content by cropping or cutting a portion of the original video, is the second operation [66].

When multiple-resolution videos are combined to create a counterfeit video, a **double quantization effect** can be used to detect the forgery. Figure 7 shows the details [53]. published a strategy to detect the double quantization video phenomenon in 2009. Later in 2015, the authors in [58] proposed an **object-based model** for video forgery detection, achieving an accuracy of 83.37%. [68] developed a **temporal noise correlation-based copy-and-paste object forgery detection technique** as well as a more **statistical categorization method** that uses the **Bayesian classifier and a Gaussian mixture model (GMM)** [54]. applied a **motion-compensated edge artifact (MCEA) differential** between adjoining frames to detect counterfeiting and examined whether or not there could be any spikes in the Fourier transform domain [6]. extracted **compressed noise** from the spatial domain using an enhanced **Huber Markov random field (HMRF)**. The noise's transition probability matrices were employed as variables to classify a video [73]. In summary, researchers employ a variety of strategies for detecting intra-frame forgeries. Applying the **adaptive threshold selection method**, the authors of [74] obtained an intra-frame forgery detection accuracy of up to 98%. This opens up a slew of possibilities for realistic implementations to identify this form of forgery.

---
### Challenges in Forgery

Although video forensics has advanced significantly over the last decade, some issues persist that depend on basic requirements [67, 70].

#### General Challenges

* **Compression:** As the compression ratio increases, the accuracy of numerous known methodologies for detecting forgeries decreases. It is also influenced by the modification of video bit rates as well as the quantization of the scale ratio. Video compression artifacts often hinder the efficacy of the detection method. Two challenges that must be addressed are video re-compression using the same encoding options and the detection of forgery in highly compressed videos [59].
* **GOP's [Group of Picture] Structure:** Today's most widely used video encoders, such as H.264/AVC, offer an adaptable GOP structure that allows for the GOP size to grow by at least 250 frames when the content of the video changes. Many of the techniques discussed here are effective for GOPs with a constant structural size, and a few of them are also effective at detecting forgeries in films with variable GOP structures. Still, most of them are unable to detect the deletion of a whole GOP or many GOPs [6].
* **Noise:** The researcher has struggled to build a new technique with new types of noise, as video noise has become more common over the preceding 15 years. Additionally, it was found that the degree of distortion in the video affects the efficacy of the detection system [14].
* **Video Background:** Several freshly proposed forgery detection algorithms can identify fraud in a static background video (but not in a dynamic or moving background video). Researchers face an extra problem, however, because few algorithms for detecting fraud in movies with shifting backgrounds have been developed [1].
* **Video frame count:** To detect inter-frame forgeries, most existing approaches focus on the number of images inserted, duplicated, or deleted. Furthermore, when the video frame count falls below a certain threshold, these methods cannot be used to detect video forgeries [44].
* **Video quality issues:** All data is collected as evidence elements for video forensic examination using CCTV footage, mobile video clips, and so on. All these types of video sources may not fulfill the quality standards needed for effective video analysis. Even after re-scaling the video or image, minute details cannot be identified from inferior footage [68, 70]. Low-resolution content has limited improvement possibilities [52], resulting in video forensic analysis issues. Furthermore, if the movie contains low-resolution footage, brightness may tamper with the analysis. On the other hand, if there are two CCTV footage from different locations, the video may be too bright or underexposed, and the investigator must manually adjust the brightness of each video to match more information. This necessitates the use of a fully qualified resource to work with such videos for evaluation. In video forensics, factors such as the camera's point of view, the motion of a subject or object, and the number of visible features all have a substantial impact on the analysis [29].

#### Limited Dataset Challenges

Forensic identification is one of the issues; the evidence photographs are verified and compared to the images in the database due to the limitations of heterogeneous face recognition. Forensic investigators employ forensic identification techniques to supplement forensic investigations by speculating on a common source between trace evidence and a known reference sample [53, 58-61]. Investigators face a hurdle in obtaining appropriate reference samples to infer from evidence samples.

When using models with deep learning (DL), the dataset is split into both training and testing instances. Many datasets might be blended for video forensics to obtain a specified set of video frames. However, the number of free-to-use open datasets is limited, and these can belong to distinct classes, resulting in lower accuracy when using DL approaches [65, 70, 71]. As a result, the primary limitation of the present approaches discussed in previous studies is the paucity of fake video datasets for comparison with experimental research. In the literature, only a few datasets have been assessed [75].

#### Data Resource Analysis Challenges

Several other sources have been able to gather video evidence, resulting in vast amounts of data. This volume of data can strain resources [9, 29, 63]. Due to resource constraints, acquiring and analyzing forensic media takes time [76]. Additional staff may be required to analyze large amounts of data. It is also challenging to find qualified technologists for video forgery analysis [2, 5, 12]. Videos have numerous aspects, but it becomes difficult when the data is vast in size, and it becomes difficult to retain data for analysis in forensics due to a lack of sufficient hardware resources [39].

#### Deep Learning/Machine Learning Is Used

So far, just a few methodologies based on the techniques of machine learning, specifically DL, are being devised. The authors have numerous opportunities to experiment with various ML/DL models to detect inter-frame and intra-frame fraud in films. In the realm of video forgery detection, researchers are encouraged to develop an automated forgery detection strategy using ML and DL algorithms [14, 54]. However, the comprehension of DL theory still has to be clarified in order to calculate the ideal number of layers or estimate convolutional, recurrent, and pooled layers [55, 57, 59]. This lack of awareness makes the use of DL methodologies in video forgery investigations difficult. DL multi-model approaches are used in video forensics to improve object categorization. Even after some outstanding breakthroughs by DL approaches, there is still room for development in areas such as dynamic experience or occlusion in video. DL variations are widely employed; however, in the Digital Verification System (DVS), age, 3D recovery, and real-time processing complexity are issues that require a quick solution [31, 37, 39].

#### Technical Challenges

Multi-model DL approaches are used to improve object categorization in video forensics. Even after achieving several exceptional results with DL approaches, there is still a need for further advancements in this field, particularly in dynamic backgrounds. Technical issues, such as **encryption, steganography, and diverse media forms**, as well as analysis, are also significant obstacles to examining the validity of evidence [32, 36, 38]. Many encryption programs are readily available, making it less difficult for the perpetrator to conceal incriminating data. Data decryption is a difficulty for forensic investigators [30, 42, 46]. Criminals are also adopting advanced steganography techniques to hide data, posing a problem for video forensic scientists [28, 47].

#### Real-Time Processing (Computational Time)

The primary goal of scientists is to reduce the time that it takes to recognize and find forgeries in video [67]. As a result of technological advancements, powerful tools to support the DL architecture are now available, enabling real-time processing [26]. conceived and constructed a deep learning model with real-time video handling functionality. Many strategies were implemented to reduce processing costs or filter size. Changing the time complexity of DL methods and removing all redundant computations in the propagation direction (forward and backward) were required. Furthermore, Deep Neural Network (DNN) GPU-based functionality outperforms the simple model of these ML methods [70, 71]. Still, the existing literature lags when it comes to real-time video processing, which could pose a problem for future DL forensic efforts [26-29].

#### Legal Challenges

Investigators may face privacy issues when utilizing video analysis for forensics investigations. The video evidence gathered should be evaluated without invading the privacy of the victim or the organization [23, 27]. Another legal barrier for investigators is adhering to established rules for document submission and analysis format [24, 34]. Forensic investigators must also deal with a variety of administrative concerns [42, 45, 46]. In a court of law, the legal status of video forensic procedures and tools used to capture and analyze data is becoming an issue [62, 65, 70]. When handling sensitive data, ethical concerns can arise [53, 57, 58, 67].

---
### Datasets

Investigators may face privacy issues when utilizing video analysis for forensics investigations. The video evidence gathered should be evaluated without invading the privacy of the victim or the organization. Another legal barrier for investigators is adhering to established rules for document submission and analysis format. Forensic investigators must also address a range of administrative concerns. In a court of law, the legal status of video forensic procedures and tools used to capture and analyze data is becoming an issue. When handling sensitive data, ethical concerns can arise.

#### Images

There are some datasets containing modified images. Some of them tend to be rather old and, as a result, outdated, while others have significant issues. Recent articles that rely on unsuitable datasets and depict them as tough testbeds are surprising.

One of the first datasets made free to the forensics community was the **Columbia (color) dataset**, which was presented in 2006. It consists of 180 faked photos with splicing. Despite its strengths, it has some serious flaws:
* It is impractical because the forgery is evident.
* The spliced section has no postprocessing.
* Only uncompressed photos are present.
* Just four lenses are used to capture both the host and spliced images, and the spliced regions.

Furthermore, considering that both zones are derived from immaculate data, it is unclear how to characterize the areas that are faked. As a result, this dataset is not suitable for use in both the training and testing phases, nor should it be used for fine-tuning, as overly optimistic results would be observed.

The **Casia dataset**, which was suggested in 2013 [77], is also widely used. Splicing has sharp borders in the initial version (v1) and is easily recognizable. The second version (v2), on the other hand, is more accurate, and the introduced elements are postprocessed to fit the picture better. Nonetheless, it has considerable polarization, as seen in [78]. Images that have been tampered with and those that have not are JPEG-limited, with differing quality factors (the former at a higher quality). As a result, a classifier that learns to differentiate between tampered and clean photographs may instead be trained on their distinct processing histories, operating excellently on samples from a certain dataset but poorly on new, unrelated images.

Another dataset containing splicing is **DSO-1** [79], a subset of the IEEE Photo Forensics contest (sadly, the original datasets generated for the contest are no longer available, and the organizers never provided the ground realities). The alterations are done with great care here, and most of them seem believable. Images are stored in the original PNG format, but most of them have been previously compressed as JPEGs. Minor issues include fixed image resolution and insufficient information about the dataset's construction, such as the number of cameras used, which could aid in evaluating the results.

The **realistic tampering dataset** offered by Korus in [80] contains forgeries of various types. The modified photographs, which are all uncompressed, appear incredibly realistic, despite being a small number. The collection also includes the PRNU characteristics of the four digital cameras used to capture all of the photos, allowing sensor-based algorithms to be applied.

The **Wild Web Dataset** [80] is a set of real-world internet cases. As a result, there is no verified information on the alterations. Still, the writers went to great lengths to collect several versions of the same photographs and derive relevant ground truths.

Many datasets have been specifically provided for detecting **copy-move fraud** [81-85], the most researched type of forgery in the literature. Some of them are designed to test copy-move methods by performing various actions on the duplicated object, such as rotation, resizing, and lighting modification. However, as the distance between an object's statistical qualities and the characteristics of the background grows, it becomes more visible for approaches relying on camera artifacts.

A realistic dataset for testing **double JPEG compression** is also available [73]. In contrast, a synthetic dataset of standard and dual JPEG compressed blocks, comprising 1,120 quantization tables, was recently provided for training deep networks [86]. Table 2 shows the datasets that contain manipulated images.

| Dataset | References | Manipulations | Images size |
| :--- | :--- | :--- | :--- |
| Columbia (color) | [78] | Splicing (unrealistic) | $128\times128$ |
| Casia v1 | [77] | Splicing, copy-move | $374\times256$ |
| Casia v2 | [77] | Splicing, copy-move | $320\times240-800\times600$ |
| DSO-1 | [79] | Splicing | $2048\times1536$ |
| Wild Web | [87] | Real-world cases | $72\times45-3000\times2222$ |
| FAU | [82] | Copy-move | $2362\times1581-3888\times2592$ |
| MICC F220 | [81] | Copy-move | 722 x $480-800\times600$ |
| MICC F2000 | [81] | Copy-move | $2048\times1536$ |
| GRIP | [83] | Copy-move | $1024\times768$ |
| CoMoFOD | [84] | Copy-move | $512\times512-3000\times2000$ |
| COVERAGE | [85] | Copy-move | $400\times486$ |
| DEFACTO | [88] | Various | $240\times320-640\times640$ |
| RTD (Korus) | [80] | Splicing, copy-move | $1920\times1080$ |
| NC2016 | [83] | Splicing, copy-move, removal | $500\times500-5.616\times3.744$ |
| NC2017 | [83] | Various | $160\times120-8000\times5320$ |
| FaceSwap | [19] | Face swapping | 450 x $338-7360\times4912$ |
| MFC2018 | [83] | Various | $128\times104-7952\times5304$ |
| PS-Battles | [89] | Various | $130\times60-10,000\times8558$ |
| MFC2019 | [90] | Various | $160\times120-2624\times19,680$ |
| GAN collection | [76] | GAN generated | $256\times256-1024\times1024$ |

#### Video

Table 3 lists the datasets that contain manipulated videos. Several of them can be used for video experiments, but the other have been quickly increasing in the past year. Creating good-quality realistic forged videos with ordinary editing tools is time-consuming; thus, only a few tiny datasets comprising classic manipulations, such as copy-moves and splicing, are available online [83, 85]. Video altered with Al-based technologies can be found in many more and much larger datasets [75, 86, 91-96].

| Dataset | References | Manipulations | Images size |
| :--- | :--- | :--- | :--- |
| DF-TIMIT | [91] | Deepfake | $64\times64-128\times128$ |
| FFW | [86] | Splicing, CGI, Deepfake | 480p, 720p, 1080p |
| FVC-2018 | [97] | Real-word cases | Various |
| FaceForensics++ | [92] | Deepfake, CG-manipulations | 480p, 720p, 1080p |
| DDD | [93] | Deepfake | 1080p |
| Celeb-DF | [94] | Deepfake | various |
| DFDC-preview | [75] | Deepfake | 180p - 2160p |
| DFDC | [95] | Deepfake | 240p - 2160p |
| DeeperForensics-1.0 | [96] | Deepfake | 1080p |

[91] presents **DF-TIMIT**, a face-swapping video dataset comprised of 620 deepfake films generated using a GAN-based technique. The initial data were obtained from a database that contained ten films for each of the 43 subjects. Sixteen pairings of subjects were hand-picked from the database to create recordings with faces changed from topic one to topic two and vice versa, yielding both lower-quality and excellent-quality videos.

Instead [86], suggests the **Fake Face in the Wild Dataset, FFW**, which contains just 150 modified films but demonstrates a wide range of methods, including editing and CG faces, using both manual and entirely automatic procedures. Finally, in [2], a dataset was compiled that included 200 modified web videos and 180 real ones. An expanded version of this dataset includes web-based near-duplicates.

**Face Forensics++**, the first large dataset containing automatically modified faces, was proposed in [92]. It includes 1,000 original films retrieved from the YouTube-8 M dataset [91] as well as 4,000 modified videos created using four distinct alteration tools. Two of these models depend on computer graphics, two on deep learning algorithms, two perform emotion alterations, and two execute face switching. The dataset is available in both original and H.264 compressed formats, with two different levels of quality, to encourage the development of compression-resistant techniques. Google and Jigsaw have added 3,000 more modified movies to the dataset, which were generated ad hoc using 28 actors [86].

Additionally, **Celeb-DF**, the latest deepfake video dataset, was introduced in [94]. It consists of 5,639 modified videos, with the actual videos based on open-source YouTube images of 59 celebrities of various categories, ages, and nationalities. Using an enhanced deepfake synthesis method, forged films are created by swapping faces for each of the 59 subjects.

Instead [75], describes the first version of the dataset employed for the **Facebook Deepfake Detection Challenge (DFDC)**. It is composed of 4,113 deepfake videos generated by two distinct synthesizing algorithms, utilizing 1,131 original footage clips featuring 66 actors. The final version of the dataset produced for the Kaggle competition [95] (which began in December 2019) is substantially larger. It includes 100,000 altered videos as well as 19,000 immaculate videos.

A fairly new dataset has been developed in [96], consisting of 10,000 false videos created using 100 actors and employing seven perturbations, such as color saturation, blurring, and compression, with varied settings for a total of 35 alternative post-processing methods to better mimic a real scenario.

---
### Conclusion and Future Work

Until about twenty years ago, only a small group of people in fields like law enforcement, intelligence, and private investigations had any real use for the discipline of multimedia forgery analysis. Both the offensive and defensive strategies were artisanal, requiring a great deal of time and effort to perfect. Scientists from a wide variety of fields are actively contributing to the development of multi-media forgery analysis today, and substantial funding is being allocated to large-scale research programs.

Artificial intelligence has largely changed these rules. High-quality fakes now seem to come out from an assembly line, calling for an extraordinary effort on the part of both scientists and policymakers. In fact, today's multimedia forensics is in full development; major agencies are funding large research initiatives, and scientists from many different fields are actively contributing, with rapid advances in ideas and tools. It isn't easy to predict whether such efforts will be able to ensure information integrity in the future or if some forms of active protection will become necessary.

Numerous strategies have been employed to identify inter-frame forgeries. The majority of studies have utilized the **REWIND dataset** for experimental purposes. For this kind of counterfeit detection, the optical path of images has been a well-liked method among researchers. On the REWIND dataset, this approach has an accuracy of 89%.

This paper studies various current technologies and provides future directions and challenges in multi-media forensics. It introduces image and video forensics along with its applications and existing datasets. Anti-forensics, compression forensics methods, and deepfake detection in multi-media have been further discussed as potential future developments. The traditional benchmark datasets for multi-media (images and video) forgeries have also been investigated.

---
### Declarations

**Conflict of interest** The authors declare that there is no conflict of interest regarding the publication of this paper.

**Ethical Approval** This article does not contain any studies with human participants or animals performed by any of the authors.

---
### References

*(References omitted as per instructions)*