# AiGen-FoodReview 

[![Conference](https://img.shields.io/badge/ICWSM-2024-blue)](https://www.icwsm.org/2024/)

This repository contains the code and data associated with the paper **AiGen-FoodReview: A Multimodal Dataset of Machine-Generated Restaurant Reviews and Images on Social Media**, which has been accepted at the [International Conference on Web and Social Media (ICWSM) 2024](https://www.icwsm.org/2024/).
Paper text is available at [link](https://ojs.aaai.org/index.php/ICWSM/article/view/31437).
The full dataset is also available on Zenodo at [link](https://zenodo.org/records/10511456).

AiGen-FoodReview is a multimodal dataset of machine-generated fake restaurant reviews and images.
The dataset is divided into 3 splits: training (60%), validation (20%), and test (20%), named train.csv, val.csv, and test.csv, respectively. 

## Abstract
Online reviews in the form of user-generated content (UGC) significantly impact consumer decision-making. However, the pervasive issue of not only human fake content but also machine-generated content challenges UGC's reliability. Recent advances in Large Language Models (LLMs) may pave the way to fabricate indistinguishable fake generated content at a much lower cost. Leveraging OpenAI's GPT-4-Turbo and DALL-E-2 models, we craft AiGen-FoodReview, a multimodal dataset of 20,144 restaurant review-image pairs divided into authentic and machine-generated. We explore unimodal and multimodal detection models, achieving 99.80% multimodal accuracy with FLAVA. We use attributes from readability and photographic theories to score reviews and images, respectively, demonstrating their utility as handcrafted features in scalable and interpretable detection models with comparable performance. This paper contributes by open-sourcing the dataset and releasing fake review detectors, recommending its use in unimodal and multimodal fake review detection tasks, and evaluating linguistic and visual features in synthetic versus authentic data.


## Data Generation Methodology
![alt text](assets/method.jpeg)

## Variable Description
Images are available to download at [link](https://drive.google.com/file/d/1FzBIklsUkNaBKdCWvjbeb3h4PH1zUI3Q/view?usp=sharing).
Each image is named "ID.jpg", mapping to each review. 

Features description: 

- ID: Unique identifier. It maps to each review, either generated or authentic, and to each image. 
- text: Review text.
- label: Binary label indicating the class (0=authentic, 1=machine-generated).
- automated_readability_index: Approximate US grade level needed to comprehend the text.
- difficult_words: Number of difficult words from Dale-Chall world list.
- flesch_reading_ease: Score on a scale from 0 to 100, with higher scores indicating easier readability.
- gunning_fog: Years of formal education a person needs to understand a text easily.
- words_per_sentence: Average number of words per sentence. 
- reading_time: Reading time.
- ppl: Perplexity score from zero-shot GPTNeo 125M. 
- bright: Brightness. Average of V of the HSV image representation.
- sat: Saturation. Color intensity and purity of an image. Average of S of the HSV image representation.
- clar: Clarity. Well-defined objects in space. % of normalized V pixels that exceed 0.7 of HSV.
- cont: Contrast. Spread of illumination. Standard deviation of V of the HSV image representation.
- warm: Warmth. Warm colors: from red to yellow. % of H<60 or $>$ than 220 of HSV.
- colorf: Colorfulness. Departure from a grey-scale image.
- sd: Size difference. Difference in the number of pixels between the figure and the ground
- cd: Color difference. Difference of Euclidian distance between the figure and ground (RGB vectors).
- td: Texture difference. Absolute difference between the foreground and background edge density.
- diag_dom: Diagonal dominance. Manhattan distance between salient region and each diagonal.
- rot: Rule of thirds. Minimum distance between center of salient region and each of the four intersection points.
- hpvb: Horizontal physical visual balance. Split image horizontally. Horizontal physical symmetry (mirroring).
- vpvb: Vertical physical visual balance. Split image vertically. Vertical physical symmetry (mirroring)
- hcvb: Horizontal color visual balance. Split image horizontally. Horizontal mirrored Euclidean cross-pixels distance.
- vcvb: Vertical color visual balance. Split image vertically. Vertical mirrored Euclidean cross-pixels distance.

Textual features were mined using [textstat](https://pypi.org/project/textstat/), while an explanation of viusal feautures can be found at [Gambetti&Han2022](https://www.emerald.com/insight/content/doi/10.1108/IJCHM-09-2021-1206/full/html).

## Training and Evaluation
Training and evaluation python scripts are available for text (BERT, GPTNeo), vision (ViT, ResNet), and multimodal models (FLAVA, CLIP). Each script comes with a CONFIG section in the beginning, with path and configuration parameters to be inserted. Optimized weights for each model are accessible at this [link](https://drive.google.com/file/d/1aYz5G6z9BL7mv9dL-JV2iay25EuKbgqn/view?usp=sharing). 

## Authors
- [Alessandro Gambetti](https://scholar.google.com/citations?user=F0MfEx8AAAAJ&hl=en)
- [Qiwei Han](https://scholar.google.com/citations?user=koBWI88AAAAJ&hl=en)

## Cite Us
```bibtex
@inproceedings{gambetti2024aigen,
  title={AiGen-FoodReview: A Multimodal Dataset of Machine-Generated Restaurant Reviews and Images on Social Media},
  author={Gambetti, Alessandro and Han, Qiwei},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={18},
  pages={1935--1945},
  year={2024}
}
```