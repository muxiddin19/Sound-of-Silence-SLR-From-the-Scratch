# Sound-of-Silence-Sign-Language-Recognition-From-the-Scratch
### This is repo for our paper submitted to IDE2025
![image](https://github.com/user-attachments/assets/d7f9e6bc-138a-4e15-9f65-9ff8e7d995d4)
 ## Abstract
 In this paper, we revolutionize the landscape of sign
 language recognition (SLR) with direction vector-based cutting
edge keypoint data vectorization techniques and a powerful mul
timodal deep learning (DL) approach. R-tree indexing employed
 our innovative vectorization method, simplifying and stream
lining SLR data, dramatically boosting data comprehension,
 storage, and computation efficiency. By leveraging a multistream
 architecture that captures multiple dimensions of sign language,
 we not only push the boundaries of interpretability but also
 deliver enhanced recognition accuracy. We address challenges
 like outliers in keypoint data and complex gestures through
 careful data preprocessing and specialized training strategies,
 improving robustness to edge cases such as occluded hand shapes
 and fast motions. By reducing the dimensionality of keypoint
 data, we can minimize the computational cost while maintaining
 high accuracy, which is crucial for real-time applications where
 latency and computational overhead are critical factors. This
 breakthrough sets a new benchmark for SLR systems, opening
 doors to real-time applications and expanding into previously
 untapped modalities. With the promise of further advancements,
 our work paves the way for the practical and widespread
 adoption of explainable and high-performance SLR technologies.
![image](https://github.com/user-attachments/assets/be2eb6ef-4a01-470d-9dbe-96d7c65b44c2)


## Dependencies
- absl-py==0.9.0
- numpy==1.18.1
- oauthlib==3.1.0
- omegaconf==2.1.1
- opencv-python==4.4.0.46
- pandas==1.0.3
- pillow==8.4.0
- portalocker==1.5.2
- protobuf==3.19.4
- py==1.8.1
- pyarrow==3.0.0
- pydot==1.4.2
- python-dateutil==2.8.1
- scikit-learn==0.22.2.post1
- tensorboard-data-server==0.6.1
- tensorboard-plugin-wit==1.8.0
- tensorflow==2.2.0
- tensorflow-estimator==2.2.0
- tokenizers==0.10.3
- torch==1.9.0
- torch-tb-profiler==0.2.1
- torchtext==0.5.0
- torchvision==0.10.0
- tqdm==4.40.2
- transformers==4.11.3

## Data Preparation
### Exploited Datasets
- Phoenix-2014: A German sign language dataset with 1081 glosses.
- Phoenix-2014T: An extension of Phoenix-2014, with 1066 glosses and 2887 German text entries.
- CSL-Daily: A large-scale Chinese sign language dataset with 2000 glosses and 2343 text entries.
### Video Data


### Keypoint Data
![image](https://github.com/user-attachments/assets/8b79934d-dd96-472f-8103-0349f2f6a205)


### R-Tree indexing
The R-Tree was applied to 2D human keypoints 
to enhance the efficiency and robustness of keypoint data retrieval in SLR.
 For each frame in a video, keypoints representing body parts (e.g., hands, face) are extracted. 
These keypoints are typically in the format (x, y, c), where x and y are the spatial coordinates, and c is the confidence score.
The R-Tree divides and indexes multi-dimensional data into Minimum Bounding Rectangles (MBRs). Each keypoint, defined by its (x, y, c) coordinates, is inserted into the R-Tree with a unique identifier. 
This allows for rapid and efficient proximity searches.



![image](https://github.com/user-attachments/assets/0a5ac44d-1890-436d-8176-96aae69c7e01)

### Doppler Data
The number of input frames and output frames is consistent, 
while it outputs temporal motion features.
![image](https://github.com/user-attachments/assets/21c966de-ed89-450f-ba0d-22c4037ef9d7)



### Vector Data
![image](https://github.com/user-attachments/assets/548586dd-ede9-48f7-baa0-0de65fb5be38)

Left: Direction vector with eight different directions, plus zero (0, 0) vector, the initial point. 
Right: Visualize the vectorization approach with a direction vector using only one keypoint.
![image](https://github.com/user-attachments/assets/d9119f2b-c201-4c65-97a8-112205b99bbe)

![image](https://github.com/user-attachments/assets/5b0ad68f-39a1-47ba-84e3-45fdba1f0b33)
