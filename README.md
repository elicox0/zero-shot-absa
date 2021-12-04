# zero-shot-absa

## About
The goal of this project is to accomplish aspect-based sentiment analysis without dependence on the severely limited training data available - that is, the task of aspect-based sentiment analysis is not explicitly supervised, an approach known as “zero-shot learning”. Sentiment analysis has already been used extensively in industry for things such as customer feedback; however, a model such as the one I am proposing would be able to identify topics in a document and also identify the sentiment of the author toward (or associated with) each topic, which allows for detection of much more specific feedback or commentary than simple sentiment analysis. 

## Details
There will be three models in the project; the first, m1, will use Latent Dirichlet Allocation to find topics in documents, implemented through gensim. The second, m2, is a zero-shot learning text classification model, available at Hugging Face, which I plan to fine-tune on output of the LDA model on various tweets and reviews. The final piece, m3, is the sentiment intensity analyzer available from NLTK’s vader module. The architecture is as follows: m1 will generate a list of topics for each document in the dataset. I will then create a mapping T from each document to the corresponding list of topics. It would be nice to have labeled data here that, given the output T(doc), supplies the human-generated topic name. Since that isn’t available, the zero-shot text classifier from Hugging Face will be used to generate a topic name, which exists only to interpret the output. Then for each topic t in T, we search the document for all sentences containing at least one word in t and use NLTK to compute the average sentiment score of each of these sentences. We then return, as the model output, the dictionary with all topic names found in the document as keys and the average sentiment from NLTK as the values.

## Dependencies
- `scikit-learn`
- `gensim`
- `NLTK`
- `huggingface.ai`

## Data
The data this project will be trained on come from Twitter and Yelp. With access to the Twitter API through a developer account, one can create a large corpus from tweets. Yelp has very relevant data for this task available at https://www.yelp.com/dataset. I will train / fine-tune each model twice, once for Twitter and once for Yelp, on a training set generated by scikit-learn.

Labeled data for testing are available at https://europe.naverlabs.com/Research/Natural-Language-Processing/Aspect-Based-Sentiment-Analysis-Dataset/ . These data are very straightforward to use, as they have annotations of topics and the associated sentiment scores for each sentence.