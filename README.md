# Assignment 4

---

The following was made for reproduceability.

Shows that NB doesn't work for autoregressive document completion, but performs exceptionally well and extremely quickly for jobs such as spam filtering with minimal feature engineering. 

### Specifications

---

- **Experiment 1**: NB vs 1DCNN for spam filtering. Word frequencies were used for both. Degredation in CNN accuracy could be due to the fact that word frequencies are not enough for the CNN to extract as much meaningful information as NB. 
- **Experiment 2**: Showcases complete degredation of NB when it comes to autoregressive text generation. Mistral-7B-Instruct-v.2 was used for comparison.

---

####**Spam Dataset**: [here](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset).

####**Huggingface Mistral-7B-Instruct-v.2**: [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).