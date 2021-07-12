# Project Overview: Medical Intent Detector Using BERT

## Project Highlights

I built a multi-class classifier using BERT from Transformers that can identify common medical symptoms based on descriptive text. For example, the model can predict the medical intent is "Neck pain" after parsing the text message "There is a tingling sensation in my neck." It can be applied to services such as medical chatbot. 

As for the model building process, I built the classifier with transfer learning from pre-trained BERT model, which was already trained on large corpus. For our specific task, the pre-trained BERT model was added an layer on top for classifying descriptive text to 25 intents (categories). When training started, I fine-tuned the entire pre-trained BERT model and the additional untrained classification layer. After 4 epochs of fine-tuning the model on thousands of text messages with a good selection of hyperparameters, I obtained 99.40% accuracy in the test set. See code [here](https://github.com/ensembles4612/medical_intent_detector_using_BERT/blob/master/medical_intent_detector_Using_BERT.ipynb).

## Resources

* **Dataset Used**: from Kaggle containing texts for common medical intents. https://www.kaggle.com/paultimothymooney/medical-speech-transcription-and-intent
* **Language**: Python 3 and PyTorch
* **BERT Research**: from Chris McCormick https://www.chrismccormick.ai/
* **BERT Paper**: https://arxiv.org/abs/1810.04805
* **Transformers Docs**: https://huggingface.co/transformers/
* **Transformers Repo**: https://github.com/huggingface/transformers
* **Packages Used**: tensorflow, torch, numpy, pandas , seaborn, matplotlib, google.colab, sklearn, transformers, time, datetime, random, os
* **Colab GPU Setup**: Colab -> New Notebook -> Edit -> Notebook Settings -> Hardware accelerator -> (GPU)


## What is BERT

The BERT model was proposed in BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It’s a bidirectional transformer pretrained using a combination of masked language modeling objective and next sentence prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia. It's provided by Transformers. Transformers provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch.

**Advantages of using BERT:**

* Quicker development: BERT model weights were pre-trained on large corpus so it takes much less time to train our fine-tuned model.
* Less data needed: the pre-trained weights allow us to fine-tune a specific task on a much smaller dataset than building from scratch. 
* Better results: the simple fine-tuning procedure (typically adding one fully-connected layer on top of BERT and training for a few epochs) was shown to achieve state of the art results with minimal task-specific adjustments for a wide variety of tasks: classification, language inference, semantic similarity, question answering, etc. 

## Dataset Preparation

* **Dataset:** The dataset contains 6661 examples. I used 2 columns, "phrase" and "prompt" for modeling. There are 25 prompts (intents). 
* **Train, validation and test sets split:** I split data to train(70%), validation(10%) and testset (20%) stratified by the variable "intent". After stratification, data for each intent will balanced and data for each set will be proportional to 70%, 10% and 20%. That is crucial for training and testing purposes.
* **Tokenization and input formatting**: I Prepared the input data to the correct format before training as follows:
  * tokenizing all sentences
  * padding and truncating all sentences to the same length.
  * Creating the attention masks which explicitly differentiate real tokens from [PAD] tokens. 0 or 1.
  * encoding the label "intent" to numbers. 25 intents to 25 numbers.
  * creating DataLoaders for our training, validation and test sets
  
## Model Building (BERT Transfer Learning)

I used BertForSequenceClassification, a BERT model with an added single linear layer on top for classification. As we feed input data, the entire pre-trained BERT model and the additional untrained classification layer is trained on our specific task.

**Model training:** After tuning all the hyperparameters with different values, I decided to use the hyperparameters below and ran 4 epochs for the training data. It took about 34s for each epoch. Training set accuracy increased from 37% (at 1st epoch), 93% (at 2nd epoch), 99% (at 3rd epoch), to 100% (at 4th epoch).  
```
TRAIN_BATCH_SIZE =32
VALID_BATCH_SIZE = 64
EPSILON = 1e-08
EPOCHS = 4
LEARNING_RATE = 2e-5
SEED = 1215
```

Also, I did the following before training the model:
* choosing pretained base(relatively small) BERT mdoel for sequence classification
```
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 25)
```
* using AdamW optimizer and creating the learning rate scheduler
* creating a function to calcuate the accuracy of the model

## Model Performance

* Validation set accuracy: 99.40%
* Test set accuracy: 99.40%

## Preparing for Deployment

* **Saving the model, tokenizer and labels:** I saved the BERT model with 99.40% test set accuracy along with the tokenizer and labels for medical intents used when developing the model.
* **Creating medical intent detector function and test with new sentence:**
  * Loaded the saved model, tokenizer and labels 
  * Created a medical_symptom_detector function with the loaded model, tokenizer and labels, which helps predict the medical intent of a medical text message. 
  * tested an unseen example on the detector 

## Future Work

The dataset that the model trained on contains 25 categories covering a decent number of common medical intents for classification, and yet there may be new text data that should be classified to an unseen intent when we use the detector to do predictions. Therefore, it might be interesting to train a new text-to-text model using transfer learing from other Transformers models. In this way, the intents such as neck pain and headaches will be treated simply as text instead of a class of a categorical variable.    

