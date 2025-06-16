# LLM Fine-Tuning Data

This repository contains a small question/answer dataset about mental-health topics along with notebooks demonstrating how to fine‑tune open source language models. Both notebooks apply the [QLoRA](https://arxiv.org/abs/2305.14314) technique to [Mistral 7B](https://github.com/mistralai/mistral-src) and [Microsoft's Phi‑2](https://huggingface.co/microsoft/phi-2).

## Dataset

Two JSON Lines files are provided:

- `train.jsonl` – 2,809 samples
- `validation.jsonl` – 703 samples

Each line contains an `input` field with a question or prompt and an `output` field with the corresponding response.

```json
{"input": "We don't have sex a lot. I cheat when we argue…", "output": "Hello, and thank you for your question…"}
```

## Notebooks

Two example notebooks walk through applying QLoRA to different models. They are meant to be run either locally in Jupyter or in Google Colab.

### `mistral_finetune_data.ipynb`

Demonstrates fine‑tuning the [Mistral 7B](https://github.com/mistralai/mistral-src) model.
The main steps are:

1. Install the required libraries (Transformers, PEFT, Accelerate, bitsandbytes and Datasets).
2. Load `train.jsonl` and `validation.jsonl` with `datasets.load_dataset`.
3. Tokenize the text using the Mistral tokenizer.
4. Load the 7B base model in 4‑bit precision via `bitsandbytes`.
5. Configure LoRA adapters with PEFT and apply them to the model.
6. Run `transformers.Trainer` with suitable `TrainingArguments` (batch size, learning rate, number of steps, etc.).
7. Save the resulting LoRA weights.
8. Training a 7B model with QLoRA typically requires a GPU with at least 16 GB of memory (a Colab T4 or better).

### `phi2-finetune-own-data.ipynb`

Uses the same approach for [Microsoft’s Phi‑2](https://huggingface.co/microsoft/phi-2).
The dataset variable names differ from the Mistral notebook (`notes.jsonl` and `notes_validation.jsonl`), so update the paths if you want to reuse `train.jsonl` and `validation.jsonl`. The notebook illustrates how to tweak hyperparameters for a smaller model. After training it saves a checkpoint that can be merged with the base model for inference.

Both notebooks rely on the following packages, which can be installed with:

```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q -U datasets scipy ipywidgets matplotlib
```

After installing the requirements, open the notebook in Jupyter and follow the instructions to load `train.jsonl`/`validation.jsonl` and launch training.

## Usage

These examples are intended as a starting point for experimenting with fine‑tuning. Adjust the hyperparameters and dataset as needed for your own projects.

