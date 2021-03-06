# Weighted-Label Word Embedding
The main goal this work is to investigate new methods to find mathematical options for adding Weights in word embedding models.  
Word2vec has recently been applied to a wide variety of tasks in Natural Language Processing (NLP) and Text Mining. It creates a normalized vector for each word which can only be labeled positive one or negative one. But in some cases when we want to see the effect of words being weighted like in specificity, we are looking for weighted linear combination of a group of words with real number coefficients that in the word2vec model has not been proposed.   
# Example
For example we know the Top closest Concepts to "France" (by Word Embedding proximity). **"France"** is the only Positive label here, and it essentially has the "weight"1.0 the vector, i.e. **(France, 1.0)**. We want to give each label (positive or negative) a weight when constructing the vector, for example, **(France, 1.0) + (Paris, 0.5)**.  
These codes contain the **_four_** heuristic methods developed according to the mentioned goal.  
You can find the detailed descriptions of these methods in the file **Weighted-Label-Word-Embedding report.pdf**.
# Contact
If you have any questions regarding the code, feel free to contact me on moein.khajehnejad@gmail.com
