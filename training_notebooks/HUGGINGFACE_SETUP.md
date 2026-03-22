# Hugging Face Setup Guide

To download datasets or push models to the Hub, you will need a Hugging Face account and an Access Token.

## 1. Create a Hugging Face Account
1. Go to the [Hugging Face Sign Up Page](https://huggingface.co/join).
2. Enter your email address and choose a password.
3. Complete your profile and verify your email address via the confirmation link sent to your inbox.

## 2. Generate an Access Token
You need an access token to authenticate your scripts and notebooks.
1. Log in to your Hugging Face account.
2. Click your profile picture in the top-right corner and select **Settings**.
3. On the left sidebar menu, click on **Access Tokens**.
4. Click the **New token** button.
5. Give the token a descriptive name (e.g., "Vera Pipeline").
6. Set the **Role** to **Write** (this is required if you intend to push the fine-tuned models back to your account).
7. Click **Generate a token**.
8. Copy the generated string (it will start with `hf_`). Keep this token completely private!

## 3. Requesting Access to Gated Datasets (e.g., LLM-AggreFact)
Certain datasets, such as the [`LLM-AggreFact` evaluation benchmark](https://huggingface.co/datasets/lytang/LLM-AggreFact), are **gated**, meaning you must explicitly agree to their terms before you are granted access.

1. Ensure you are logged into your Hugging Face account.
2. Navigate to the dataset's repository page on Hugging Face using your browser: [lytang/LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact).
3. You will see a prompt requesting you to agree to the repository's terms and conditions.
4. Read through the terms, check the required boxes, and click **Agree and access repository**.
5. Once access is granted (usually instantly), your Access Token will now have permissions to download this dataset programmatically via your notebooks.

## 4. Using the Token in the Notebooks
In the provided Jupyter Notebooks (e.g., `eval_vera_aggrefact_cot.ipynb` and `vera_decomposer_finetune_cot.ipynb`), there are cells that require this token to push the finished model to the Hub or download gated datasets.

Find the authentication lines that look like this:
```python
token = None,
```

Replace `None` with your actual token string inside quotation marks:
```python
token = "hf_your_actual_token_string_here",
```
