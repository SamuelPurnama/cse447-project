# Project plan

## 1. Dataset

### Data Acquisition Strategy

We will train a character-level next-character prediction model using a two-source dataset strategy that balances (1) realistic everyday communication patterns with (2) coverage of technical/space terminology relevant to astronauts.

**Primary Source (Generalization):**

- **Dataset:** OpenSubtitles (OPUS)
- **Rationale:** The system is intended for private, natural language messaging. Subtitle text closely resembles short conversational messages: it contains informal phrasing, common expressions, questions/answers, contractions, and frequent punctuation, all of which are important for character-level prediction in real communication.
- **Filtering plan:** We will filter or downweight genres that are likely to introduce niche vocabulary that does not represent typical messaging (e.g., heavy sci-fi jargon). The goal is to keep the dataset focused on communication patterns rather than specialized or imaginative terminology.

**Secondary Source (Space / Technical Vocabulary Coverage):**

- **Dataset:** Wikipedia text (with emphasis on space-related pages or categories)
- **Rationale:** Astronauts may communicate using technical terms (equipment names, procedures, orbital concepts). Wikipedia provides broad coverage and well-formed text, helping the model learn spelling and character sequences of domain-specific words (e.g., "module", "payload", "telemetry", "oxygen", "airlock"), while still being general enough to not overfit to one writing style.
- **Curation plan:** We will prioritize space-related articles (or mix a smaller subset of space-related content into a larger general Wikipedia sample) so we gain terminology coverage without overwhelming the conversational style learned from subtitles.

## 2. Method

We propose a phased approach, beginning with a statistical baseline to establish performance benchmarks, followed by a transition to a deep neural network if the baseline fails to get good performance.

### Phase 1: Statistical Baseline (N-Gram Model)

Our initial approach uses a classic Markov chain model (N-Gram) to predict the next character based solely on the frequency of preceding characters.

- **Logic:** We assume the Markov property, where the probability of the next character depends only on the previous (n-1) characters.
- **Implementation:**
  - **Language:** Python.
  - **Libraries:** We will implement this from scratch using standard dictionaries or collections. Counter to maintain maximum control over the smoothing techniques, or utilize NLTK for efficient counting.
  - **Hyperparameters:** We will experiment with N=3 and N=5.
- **Evaluation:**
  - We will measure Perplexity and strict Accuracy on the validation set.

### Phase 2: Deep Learning Approach (Transformer)

If the N-Gram model proves insufficient, we will implement a decoder-only Transformer, modeled after the GPT architecture but scaled down for character-level tasks.

- **Model Architecture:**
  - **Type:** NanoGPT (a lightweight PyTorch implementation of GPT) or a ByT5-inspired encoder-decoder, or other open-source per-trained models if needed
  - **Input:** Character indices fed into a learnable embedding layer.
  - **Core Mechanism:** Multi-Head Causal Self-Attention. This allows the model to look back at the entire context window (up to 256 or 512 characters) to calculate the probability of the next character, solving the "short memory" issue of N-Grams.
  - **Output:** A softmax layer over the vocabulary size, producing a probability distribution for the next character.
- **Implementation Details:**
  - **Framework:** PyTorch. This provides the flexibility to define custom character-level embeddings that off-the-shelf LLM libraries often obscure.
  - **Training Objective:** We will minimize the Cross-Entropy Loss between the predicted distribution and the actual next character.
  - **Optimization:** We will use the AdamW optimizer

