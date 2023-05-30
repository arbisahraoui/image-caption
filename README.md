# image-captioning

**Image caption generation**, also known as image captioning, refers to the task of producing a descriptive text that depicts the content of an image. This process combines the fields of Natural Language Processing and Computer Vision to generate accurate and meaningful captions for images.

## 1. The dataset
* **Flickr8k_images**: Contains 8092 photographs in JPEG format.
* **Flickr8k_text**: Contains a number of files containing different sources of descriptions for the photographs.

The dataset has a pre-defined training dataset (6,000 images), development dataset (1,000 images), and test dataset (1,000 images).

## 2. Training and validation results

**BLEU Scores on test dataset**

| BLEU | BLEU Score |
| ------ | -------- |
| BLEU-1 | 0.456858 |
| BLEU-2 | 0.280111 |
| BLEU-3 | 0.207527 |
| BLEU-4 | 0.103957 |


## 3. Test results

Generated captions for some examples.

#### Example 1:
![Screenshot from 2023-05-30 15-45-54](https://github.com/arbisahraoui/image-caption-generation/assets/134800339/865d2156-64ed-4d48-bb4f-dd104dacd35c) 

#### Example 2:
![Screenshot from 2023-05-30 15-47-07](https://github.com/arbisahraoui/image-caption-generation/assets/134800339/ad490946-09be-4202-9b42-56724216181c)


## 4. Limitations
While 6000 images can be sufficient for initial exploration, it may have limitations in terms of coverage, diversity, and generalization. Some problems:

* Difficulty handling uncommon scenarios
* Increased sensitivity to noise

Caption generation models often require large amounts of diverse training data to capture the complexity and variability of natural language descriptions for different images. Larger datasets provide more examples for the model to learn from, enabling it to generalize better and produce more accurate and diverse captions. So we may considere training on larger and more diverse datasets...

Also, better generalization could be reached using attention mechanism.

