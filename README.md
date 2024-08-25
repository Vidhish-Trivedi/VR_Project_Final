# Task Description And Objectives

- **Develop a Visual Question Answering (VQA) Model**: Create an advanced model that can accurately interpret and respond to questions based on the visual content of an input image. The goal is to enable the model to understand the context and details of the image and provide relevant answers to corresponding questions, enhancing the interaction between visual data and natural language processing.

- **Fine-tune Pre-trained Models**: Utilize existing pre-trained models as a foundation and refine them further to improve their performance in the VQA task.

- **Incorporate LoRA (Low-Rank Adaptation)**: Experiment with the integration of LoRA in the fine-tuning process. LoRA is a technique designed to reduce the number of trainable parameters, thereby potentially decreasing the computational resources required and speeding up the training process. We aim to explore whether LoRA can maintain or improve the model's performance while optimizing the training efficiency.

- **Performance and Training Time Comparison**: Conduct a comparison between models fine-tuned with LoRA and those fine-tuned without it. This involves evaluating various metrics such as accuracy and response time. The aim is to determine the impact of LoRA on both the quality of the model's answers and the overall training time, in order to gain insights into the benefits and trade-offs of using LoRA.

# Visual Question Answering Dataset (VQA v2)

The Visual Question Answering (VQA) v2 dataset is a large-scale benchmark dataset used to train and evaluate models designed to answer questions about images. Below are the key features and components of the VQA v2 dataset:

## Overview

- **Purpose**: The VQA v2 dataset is created to support the development and evaluation of algorithms that can understand images and answer natural language questions related to the images.
- **Data Composition**: The dataset consists of images paired with questions and answers. Each image typically has multiple questions associated with it, and each question has several ground truth answers provided by different annotators.

## Key Features

### Images
- Sourced from the COCO (Common Objects in Context) dataset.
- Contain diverse and complex scenes.

### Questions
- Open-ended questions requiring an understanding of the image content.
- Questions cover various types, including *What*, *Where*, *When*, *Why*, *Who*, and *How*.
- Designed to require different levels of reasoning.

### Annotations
- Rich annotations provide additional context, such as question types and answer confidence.

### Data Collection
- Collected via Amazon Mechanical Turk.
- First version released in October 2015.
- Second version released in 2017.

### Dataset Versions
- MS COCO (Microsoft Common Objects in Context)
- Binary Abstract
- Abstract Dataset

### Current Case Study
- Using the MS COCO training dataset.

### MS COCO Dataset Details
- Large-scale image dataset containing images of everyday objects, birds, animals, food, scenes, and humans.
- Contains 82,783 images.
- Includes 443,757 questions.
- Provides 4,437,570 answers.
- Each image has at least 3 questions.
- Each question has 10 ground truth answers from different humans.
- Data is provided in `.json` file format, including Questions, Answers, and Image IDs.

### question_type
- Type of question determined by the first few words of the question.

### multiple_choice_answer
- Most frequent ground-truth answer from ten answers.

### answers
- Each question was answered by ten subjects/humans within two-three words along with their confidence.

### answer_confidence
- Subject’s confidence in answering the question: “yes”, “no”, or “maybe”.

### answer_type
- Type of the answer. Currently three types of answers: “yes/no”, “number”, and “other”.

# Exploratory Data Analysis

- The number of question and answer pairs in the entire dataset is: **443,757**.
- On viewing the images, it is apparent that they are of different shapes.
- On analyzing the text questions, we found that most of them are written in a simple manner, but some of them have common contractions like "what's", "they're", "guy's", "dog's", etc. Some questions also have words in quotes, such as "Merry".
- The text answers are also simple, with just one word or number, or sometimes a short phrase. Some answers also contain punctuation. We resort to cleaning the questions and answers by replacing contractions with their expanded form and converting everything to lowercase. Finally, we remove punctuation.

**For Images:**

- Maximum Height: 640
- Maximum Width: 640
- Minimum Height: 51
- Minimum Width: 59
- Mean Height: 482.51
- Mean Width: 580.63
- All the images have the same channel depth of 3.
- We can infer that there is a need to resize all images to the same size before proceeding.

**For Questions:**

- The maximum character length of a question: 99
- The minimum character length of a question: 9
- The average character length of a question: 29
- The maximum word length of a question: 22
- The minimum word length of a question: 2
- The average word length of a question: 6
- Total unique words in questions: 13,332

- Questions starting with "what" commonly have answers like color names, sports names, activity names, object names, or food names.
- Questions starting with "how" mostly have answers as numbers or quantifier words like many, few, more, some, all, etc.
- Questions starting with "is" often have "yes" or "no" as answers.

**For Answers:**

- The maximum character length of an answer: 71
- The minimum character length of an answer: 1
- The average character length of an answer: 4
- The maximum word length of an answer: 18
- The minimum word length of an answer: 1
- The average word length of an answer: 1
- Total number of unique answers: 22,350
- Types of answers as per given `answer_type`: 3
- 92% of the answers are one-word answers, 5% are two-word answers, the rest vary.

# Data Preparation and Preprocessing

We now describe our data preprocessing strategy. The associated Jupyter notebook has been submitted.

1. Starting with the entire VQA dataset, we transform the JSON files for questions and annotations to CSVs, and separate the answers column (which was originally a list of ten answers per question) into multiple columns, each representing one answer.
2. We made the choice to always consider the first answer to a question as its label, i.e., the answer to the question. We then drop certain columns and keep only `image_id`, `question_id`, and `answer`. This gives us a dataframe for the annotations.
3. Next, we work on the questions data file, again dropping the `image_id` column. This gives us a dataframe for the questions.
4. The annotations dataframe and the questions dataframe are merged on the `question_id` column.
5. Now, we make another choice to keep only one-word answers, as our proposed model works like a classifier and is not a generative model. We also restrict our answer space to the top 1000 most frequently occurring one-word answers.
6. Since the dataset is large, we sample 25% of the data to fine-tune and train our model. We randomly shuffle the list of all `image_id`s and pick the first 25% of `image_id`s. Since each `image_id` has a good distribution of questions of different types, we avoid having a large number of data points from the same question type. We then filter the data to keep only the questions for which the `image_id` has been sampled.
7. To generate the answer space for the prepared data, we simply create a list of unique answers using both the validation data as well as the train data.
8. Finally, we remove rows having NULL values for the answer, if any.

# Approach to Visual Question Answering at a High Level

Our approach involves the following key steps:

1. **Text Embedding**: We employ a pretrained BERT model to encode the textual input, typically a question related to an image. The BERT model is fine-tuned to generate contextual embeddings for the input text, capturing the semantic meaning necessary for understanding the question.

2. **Image Embedding**: For image encoding, we use the Vision Transformer (ViT) model, pretrained on a large dataset. The ViT model processes the input images to produce high-dimensional feature representations that encapsulate visual information relevant to answering the question.

3. **Fusion of Embeddings**: The text and image embeddings are concatenated and passed through a fusion module. This module combines the multimodal information to create a unified representation that leverages both the visual and textual data.

4. **Answer Prediction**: The fused embeddings are input to a classifier, which maps the combined representation to a predefined answer space of approximately 1000 classes. This enables the model to predict the most likely answer to the given question based on the integrated multimodal information.

# Model Architecture

Our Visual Question Answering (VQA) model is designed using a multimodal approach that combines the capabilities of BERT for text processing and Vision Transformer (ViT) for image processing. The architecture is structured into several key components, each responsible for specific aspects of the data processing pipeline. Below, we detail each component and its role in the overall architecture.

**NOTE**: Refer to Figure-1 for pseudocode of the model, and Figure-2 for an overview of the model architecture.

```python
class MultimodalVQAModel(nn.Module):
    def __init__(
        self,
        num_labels=len(answer_space),
        intermediate_dim=512,
        pretrained_text_name='bert-base-uncased',
        pretrained_image_name='google/vit-base-patch16-224-in21k'
    ):
        ...
        
        # Text & Image Encoders
        self.text_encoder = AutoModel.from_pretrained(self.pretrained_text_name)
        self.image_encoder = AutoModel.from_pretrained(self.pretrained_image_name)

        # Fusion module
        self.fusion = nn.Sequential(
            nn.Linear(
                self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, 
                intermediate_dim
            ),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Classifier
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        ...
```

## Classification and Prediction

1. **Classifier**: The fused embeddings are input to a classifier, which is a linear layer mapping the intermediate representation to the predefined answer space. The output logits from the classifier represent the model's predictions for each class in the answer space.

2. **Cross-Entropy Loss**: During training, the model is optimized using a cross-entropy loss function, comparing the predicted logits with the ground truth labels. This loss function guides the model's learning process to minimize the discrepancy between predicted and actual answers.

## Training and Evaluation

1. **Training Process**: The model is trained on a dataset comprising questions and corresponding images, with the training process involving the optimization of both the text and image encoders alongside the fusion and classification layers.

2. **Evaluation**: Evaluation metrics include the Wu-Palmer similarity score, accuracy, and F1 score, providing a comprehensive assessment of the model's performance in the VQA task.

## The Wu & Palmer Similarity: Measuring Semantic Relations

The Wu & Palmer similarity metric serves as a tool for finding the semantic similarity between two words or phrases within a given context. It operates by analyzing the positions of the concepts represented by the words or phrases, along with their relative proximity to their Least Common Subsumer (LCS).

In a directed acyclic graph, the Least Common Subsumer refers to the deepest node that shares both considered nodes as descendants. Notably, each node is regarded as a descendant of itself within this framework.

The Wu & Palmer similarity metric has demonstrated efficiency particularly in scenarios involving single-word answers, which aligns closely with the primary focus of our task in visual question answering. However, its applicability to phrases or sentences may be limited due to its inherent design.

To facilitate the computation of the Wu & Palmer similarity score, the Natural Language Toolkit (NLTK) offers an implementation based on the WordNet taxonomy.

## LoRA: Low Rank Adaptation for Efficient Fine-Tuning

### A Quick Introduction

LoRA, short for Low-Rank Adaptation, is a technique used to efficiently fine-tune large pre-trained deep learning models. Instead of updating all the parameters of a model, LoRA introduces trainable low-rank matrices that are added to the existing weight matrices. This method significantly reduces the number of parameters that need to be adjusted during fine-tuning, making the process computationally efficient and less memory-intensive.

### Key Points of LoRA

1. **Parameter Efficiency**: LoRA reduces the number of trainable parameters by decomposing the weight update matrix into the product of two lower-rank matrices. This approach limits the degree of freedom while still capturing essential task-specific information.

2. **Memory Efficiency**: By only updating a small set of parameters, LoRA minimizes the memory footprint compared to traditional fine-tuning methods, which is particularly beneficial when working with very large models.

3. **Maintaining Performance**: Despite the reduced parameter count, models fine-tuned with LoRA often achieve performance comparable to or even better than models fine-tuned using full parameter updates.

### Adding LoRA To Our Implementation

We used Microsoft's LoRA library (available [here](https://github.com/microsoft/LoRA)) to incorporate LoRA into the fine-tuning process. The overall implementation of `MultimodalVQAModelLora` in Figure 1 is quite similar to the barebones `MultimodalVQAModel` in Figure 2. However, note that the fusion module has been slightly changed—we now use `lora.Linear` instead of `nn.Linear`. This marks the particular layer to be a "LoRA" layer, and we pass the argument `r=16` to set the rank (for decomposition) as 16. The same is repeated for the classifier layer as well.

```python
class MultimodalVQAModelLora(nn.Module):
    def __init__(
        self,
        num_labels: int = len(answer_space),
        intermediate_dim: int = 512,
        pretrained_text_name: str = 'bert-base-uncased',
        pretrained_image_name: str = 'google/vit-base-patch16-224-in21k'
    ):
       ...
        
        # Text and image encoders
        self.text_encoder = AutoModel.from_pretrained(self.pretrained_text_name)
        self.image_encoder = AutoModel.from_pretrained(self.pretrained_image_name)
        
        # Fusion module
        self.fusion = nn.Sequential(
            # r=16 is the rank to be used
            lora.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim, r=16),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Classifier
        # r=16 is the rank to be used
        self.classifier = lora.Linear(intermediate_dim, self.num_labels, r=16)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        ...
```

## Observations

We first fine-tuned a barebones (referred to as base in this subsection) `MultimodalVQAModel` on the VQA dataset, and then fine-tuned two separate instances of `MultimodalVQAModelLora` on the same dataset using rank 16 and rank 8. The following observations were made:

1. **Training Time**: 
   - Base: 199 minutes
   - LoRA (rank 8): 92 minutes
   - LoRA (rank 16): 93 minutes

2. **Evaluation Time**: 
   - Base: 6 minutes
   - LoRA (rank 8): 5.5 minutes
   - LoRA (rank 16): 5.71 minutes

3. **Evaluation Accuracy**: 
   - Base: 0.4083
   - LoRA (rank 8): 0.2271
   - LoRA (rank 16): 0.2305

4. **Evaluation F1 Score**: 
   - Base: 0.04589
   - LoRA (rank 8): 0.0005781
   - LoRA (rank 16): 0.000378

5. **Evaluation Loss**: 
   - Base: 2.099
   - LoRA (rank 8): 7.188
   - LoRA (rank 16): 7.426

6. **Training Loss**: 
   - Base: 2.23
   - LoRA (rank 8): 4.153
   - LoRA (rank 16): 4.142

7. **Overall Runtime**: 
   - Base: 4h 38m 54s
   - LoRA (rank 8): 1h 50m 24s (and a second run of 2h 16m 44s)
   - LoRA (rank 16): 1h 53m 42s

8. **Evaluation Precision**: 
   - Base: 0.24533799533799536
   - LoRA (rank 8): 0.02734375
   - LoRA (rank 16): 0.017797017797017797

9. **Evaluation Recall**: 
   - Base: 0.2462121212121212
   - LoRA (rank 8): 0.05257936507936508
   - LoRA (rank 16): 0.02904040404040404

Other details can be viewed in the interactive comparison [here](https://api.wandb.ai/links/vt-dl/pv9ob8x2), generated using *wandb*.

## Future Scope and Additional Remarks

1. **Modularity**: The proposed model allows for modularity, meaning the fusion module can be easily replaced. Some alternatives (instead of simply concatenating the image and text embeddings) could include `element-wise dot product`, `converting to the same dimension and taking a weighted sum (with learnable weights)`, or `co-attention (similar to VilBERT)`.

2. **Exploration Constraints**: Initially, we considered several possibilities mentioned above, but due to time and computation resource constraints, we could not explore all of them.

3. **Additional Resources**: 
   - Plots for various metrics are available [here (see wandb directory)](https://iiitbac-my.sharepoint.com/:f:/g/personal/vidhish_trivedi_iiitb_ac_in/EuYWgCsMWKpFr7VJtmOjxkYBpAVZMghxvLG75PHXkjmMUA?e=gHLWmW) and [here](https://api.wandb.ai/links/vt-dl/pv9ob8x2). To keep the report short, we have added these to OneDrive.
