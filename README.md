# Deep-Fake Detection Approach of GANs-based Morphing Attacks By using Benford’s Law
## Abstract
In recent years, the rapid growth of revolutionary image-synthetic techniques such as Nvidia StyleGAN has brought both chances and challenges to us. In this situation, the deep fake images based on these techniques have increased the difficulty in face recognition systems. The continuing increment of realistic fake images has brought concerns about the possible vulnerabilities in image-based security systems. Our objective is to investigate if we can use a common anomaly detection algorithm, Benford’s Law,  to detect fake-generated faces. Benford’s Law is widely used as an effective countermeasure to detecting anomalies in the first-digit probabilities of genuine numerical data. In our study, the primary method is to obtain the Discrete Cosine Transform values from the JPEG images and feed the extracted features into the Benford’s-Law-based classifier. The designated classifier will compare the empirical data with the theoretical Benford’s Law baseline by using a combination of Reyi’s, Tsallis’, and Kullback-Leibler’s divergences. The experiment tested with two datasets, one with 200 real images and 200 (General Adversarial Network) GAN-based fake images (100 labeled as easy and 100 as difficult). The result came out with a 49.5% classification accuracy, 76% in False Negative Rate, and 24% in True Positive Rate. Due to the unexpected result, we adopted a Chi-Square-based classification for further verification. After obtaining the results from the methodology used, we determined that Benford’s Law is not optimal in detecting GAN-based fake facial images. To sum it all up, Benford’s Law performs as a good measurement for real-life numerical data such as the financial and population domain, but not the same expectation for the graphical domain. One possibility of the unexpected result is that GAN attempts to minimize the difference between the samples, which makes GAN-based images hard to differentiate between what's real and fake.

## Code Installation and Configuration

To run the code files, we recommend running it on an online environment, such as Google Colab or Jupyter Notebook. In our case, we ran it using Jupyter Notebook.

The following program was built during the months of **Oct-Dec of 2023**. Running it later than this date may require some modifications to the code and dependencies.

## Image Conversion (Optional)
When planning on simulating the code, we recommend to make sure that the dataset contains JPEG-type images (Preferably 600x600). This module is given if you plan on using a different dataset and the images are in PNG format, we highly recommend you convert those images to JPEG using the [*ConvertPNGtoJPEG.ipynb*](code/ConvertPNGtoJPEG.ipynb) file. Otherwise, you do not have to run that file.

## Implementation 
Once you have your dataset prepared, or plan on using our dataset, we recommend you run the files in this order:
 - [**Benford's_Law_Divergence_Method.ipynb**](code/Benford's_Law_Divergence_Method.ipynb)
    - DCT extractor
    - First Digit Probabilities extractor
    - Classifier combining three divergences for detecting fake and real images
 - [**Chi-square Method.ipynb**](code/Chi-square-Method.ipynb)
    - Chi-Square classification implementation
## Datasets
We opted to use the following datasets from the following resources:
 - **StyleGAN3 Generated Dataset**: https://github.com/NVlabs/stylegan3 
    - Visit the referenced Github repository on how to generate your own dataset!
    - NOTE: Make sure you meet the hardware requirements from the StyleGAN git repository before using it to build your dataset.
    - FFHQ pre-trained model has been selected for our project.
 - **Kaggle Dataset**: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
    - Various other datasets could be used. 
    - Preferably datasets with dimensions 600x600 and in JPEG format
## Questions?
If you run into issues or need assistance running it, feel free to contact us via email:
 - Juan Martinez: junmartinez@mail.fresnostate.edu
 - Juan Marquez Diaz:  juanmark21@mail.fresnostate.edu
 - Yubo Zhou: yubozhou@mail.fresnostate.edu
