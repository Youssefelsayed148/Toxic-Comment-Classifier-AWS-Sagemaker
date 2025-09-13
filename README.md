# Toxic-Comment-Classifier-AWS-Sagemaker1
🚀 Toxic Comment Classifier with HuggingFace &amp; AWS SageMaker

📌 Project Overview

Online platforms often suffer from toxic and harmful comments that negatively impact user experience.
This project builds a machine learning pipeline to detect and classify toxic comments using NLP (Natural Language Processing).

We fine-tuned a DistilBERT model with the HuggingFace Transformers library on AWS SageMaker, pushed it to the Hugging Face Hub, and tested it locally via from_pretrained(). The model can also be used through the Hugging Face Inference API for easy integration.

🛠 Tech Stack

-AWS SageMaker : Training & managed ML environment
-HuggingFace Transformers : Pre-trained NLP model (DistilBERT)
-PyTorch : Deep learning backend
-Amazon S3 : Data storage
-Python : Data processing, training scripts
-Jupyter Notebooks : EDA, evaluation

⚙️ Project Pipeline

🔹 Data Preparation
- Preprocessed toxic comment dataset (train, validation, test).
- Uploaded splits to Amazon S3.

🔹 Model Training
- Fine-tuned DistilBERT using SageMaker training jobs.
- Trained on GPU instance (ml.g4dn.xlarge) for efficiency.

🔹 Evaluation
- Achieved 96.8% accuracy and 0.83 F1-score on the test set.
- Reported precision/recall for robust toxic detection.

🔹 Model Hosting
- Pushed trained model artifacts to the Hugging Face Hub.
- Reloaded locally via from_pretrained() for evaluation and testing.

🔹 Deployment (Optional)
- Accessible via the Hugging Face Inference API for real-time use cases.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
 
model_id = "Youssef-El-SaYed/toxic-comment-classifier"

# Define mapping
id2label = {0: "Non-Toxic", 1: "Toxic"}
label2id = {"Non-Toxic": 0, "Toxic": 1}

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    id2label=id2label,
    label2id=label2id
)

nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

print(nlp("You are so stupid and annoying!"))  
print(nlp("I really like your work, keep it up!"))
