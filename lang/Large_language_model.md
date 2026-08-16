---
description: A concise overview of large language models (LLMs): Transformer architecture, tokenization, pretraining and fine-tuning, key challenges…
---

 

# Large language model

From Systems Analysis Wiki

[Jump to navigation](#mw-head) [Jump to search](#searchInput) 

**A large language model** (LLM) is a machine learning model implemented with a deep neural network containing billions or more parameters and trained on extensive text corpora. In this context, "large" denotes both the number of parameters and the size of the training dataset, which in modern systems can encompass several petabytes and trillions of [tokens](https://systems-analysis.ru/eng/Token%5F%28LLM%29 "Token (LLM)"). LLMs are typically trained using self-supervised or semi-supervised approaches that involve predicting the next token in a sequence, enabling the model to learn statistical patterns in language. Increases in parameters, data volume, and computational steps have been shown to yield predictable improvements in performance, as demonstrated by _[scaling laws](https://en.wikipedia.org/wiki/Neural%5Fscaling%5Flaw)_. 

Since the introduction of [BERT](https://systems-analysis.ru/eng/BERT%5F%28language%5Fmodel%29 "BERT (language model)") (2017) and particularly [GPT](https://systems-analysis.ru/eng/GPT%5F%28OpenAI%29 "GPT (OpenAI)")\-3 (2020), large language models have become the dominant approach in natural language processing. Contemporary models, including GPT-4o, [Claude](https://systems-analysis.ru/eng/Claude%5F%28Anthropic%29 "Claude (Anthropic)") 3, [Gemini](https://systems-analysis.ru/eng/Gemini%5F%28Google%29 "Gemini (Google)") 1.5, and [LLaMA](https://systems-analysis.ru/eng/LLaMA%5F%28Meta%5FAI%29 "LLaMA (Meta AI)") 3, are capable of generating text and program code, translating, summarizing, answering questions, and constructing chains of reasoning without specialized tuning. [Multimodal](https://systems-analysis.ru/eng/Multimodal%5Flarge%5Flanguage%5Fmodels "Multimodal large language models") variants can also process images, audio, and video. Adaptation to specific applications is achieved through [fine-tuning](https://systems-analysis.ru/eng/Fine-tuning%5F%28deep%5Flearning%29 "Fine-tuning (deep learning)") or [prompt engineering](https://systems-analysis.ru/eng/Prompt%5Fengineering "Prompt engineering"), also known as [in-context learning](https://systems-analysis.ru/eng/In-Context%5FLearning "In-Context Learning"). Despite these advancements, LLMs inherit biases and inaccuracies from their training data, are susceptible to "[hallucinations](https://systems-analysis.ru/eng/LLM%5Fhallucinations "LLM hallucinations")," and demand substantial computational resources. As a result, ongoing research emphasizes behavioral alignment, corpus filtering, and the development of energy-efficient architectures. 

## Contents

* [1 Architecture](#Architecture)  
  * [1.1 Tokenization](#Tokenization)
  * [1.2 Attention Mechanism](#Attention%5FMechanism)
  * [1.3 Technical Implementation of the Attention Mechanism](#Technical%5FImplementation%5Fof%5Fthe%5FAttention%5FMechanism)  
    * [1.3.1 Query, Key, and Value](#Query,%5FKey,%5Fand%5FValue)
* [2 Training Large Language Models](#Training%5FLarge%5FLanguage%5FModels)
* [3 Problems and Limitations](#Problems%5Fand%5FLimitations)
* [4 External links](#External%5Flinks)
* [5 Literature](#Literature)

## Architecture

Modern LLMs almost exclusively use the **Transformer** architecture—a network featuring a self-attention mechanism. The Transformer model was first proposed in the 2017 paper "Attention is All You Need" by researchers at Google. 

The **Transformer** architecture is a fundamental neural network design for processing sequences, comprising two main modules: an **[encoder](https://systems-analysis.ru/eng/Encoder%5F%28Transformer%29 "Encoder (Transformer)")** (which encodes the input) and a **[decoder](https://systems-analysis.ru/eng/Decoder%5F%28Transformer%29 "Decoder (Transformer)")** (which generates the output). An input sequence is received, its vector representation ([embedding](https://systems-analysis.ru/eng/Embedding%5F%28NLP%29 "Embedding (NLP)")) is created, a positional encoding vector is added, and then the set of elements, regardless of their order in the sequence, is fed into the encoding component (parallel processing). The decoding component then receives part of this sequence and the output from the encoder. The result is a new output sequence. 

The encoding component of a Transformer consists of several identical encoder layers; the decoding component is structured similarly. The Transformer itself is a sequence of attention models that transform the original sequence of vectors into a new sequence where each element considers the context of the others. The encoder forms hidden representations of the input data, preserving information about the relationships between elements. The decoder uses these hidden representations to create a new sequence of embeddings for the output tokens. These embeddings are then used by the language model to generate the final output elements. 

Since Transformers were originally developed for tasks like machine translation, their architecture includes an **[encoder](https://systems-analysis.ru/eng/Encoder%5F%28Transformer%29 "Encoder (Transformer)")** (processing the input text, such as a source sentence) and a **[decoder](https://systems-analysis.ru/eng/Decoder%5F%28Transformer%29 "Decoder (Transformer)")** (generating the output, such as a translation). However, many language models use only the decoder part, operating in an autoregressive mode. 

Transformers are used in three main configurations, each utilizing the encoder and decoder differently and tailored to its specific set of tasks: 

* **[Encoder-only](https://systems-analysis.ru/eng/Encoder-only%5Fmodels "Encoder-only models") Transformers** (_bidirectional_) are trained to restore intentionally hidden fragments of text, making them well-suited for "understanding" tasks such as classification, fact extraction, and semantic search.
* **[Decoder-only Transformers](https://systems-analysis.ru/eng/Decoder-only%5Fmodels%5F%28architecture%29 "Decoder-only models (architecture)")** (_autoregressive_) are optimized to predict the next token and are used for tasks requiring sequential output, such as conversational agents, code completion, and creative text generation.
* **Full [encoder-decoder architectures](https://systems-analysis.ru/eng/Encoder%E2%80%93decoder%5Farchitecture "Encoder–decoder architecture")** combine both approaches: the encoder builds a representation of the entire input text, and the decoder generates the result step-by-step based on this representation. This configuration is most effective for machine translation, summarization, and question-answering systems.

### Tokenization

**[Tokenization](https://systems-analysis.ru/eng/Tokenization%5F%28NLP%29 "Tokenization (NLP)")** is a key initial step in text processing for large language models. In this stage, a continuous string of characters is broken down into individual units called **[tokens](https://systems-analysis.ru/eng/Token%5F%28LLM%29 "Token (LLM)")**. Tokenization transforms a sequence of characters into a sequence of structured elements, enabling the neural network to process it efficiently. 

From a linguistic perspective, tokenization could be seen as analogous to identifying the smallest units of language that carry independent meaning or functional weight, such as words, morphemes, or their fragments. However, a token is often just a statistically frequent sequence of characters, so it is important not to overestimate the linguistic meaningfulness of _all_ tokens. Depending on the chosen scheme, a token can be a whole word, a subword, a single character, or a special marker (e.g., start and end of sequence markers). 

Tokenization allows for: 

* Limiting the vocabulary size to a manageable scale;
* Correctly handling rare and new words;
* Ensuring a one-to-one mapping of text to a sequence of numerical identifiers.

Subword algorithms are used for text segmentation, with the most common being **Byte Pair Encoding (BPE)**, **WordPiece**, and **UnigramLM**. Each of these builds a vocabulary from the most frequent fragments in a corpus and uses it to segment any input text sequentially. 

After the tokenization stage, each text unit—a token—is converted into a numerical representation that the neural network can understand. This process involves several sequential steps: 

* _Converting tokens to identifiers_: Each token is mapped to a unique numerical index based on a pre-built token vocabulary. Textual tokens are replaced with their unique numerical identifiers (IDs). Each ID is the token's number in a pre-built dictionary, allowing the neural network to work with numbers instead of words.
* _Converting identifiers to embeddings_: For each token ID, a corresponding fixed-dimensional vector—an **[embedding](https://systems-analysis.ru/eng/Embedding%5F%28NLP%29 "Embedding (NLP)")**—is retrieved or computed. This multidimensional numerical representation replaces the ID and already contains information about the token's meaning and contextual properties. All embeddings have the same length for ease of processing.
* _Adding positional encodings_: Since the [Transformer architecture](https://systems-analysis.ru/eng/Transformer%5Farchitecture "Transformer architecture") itself does not consider the _order_ of elements, **positional encodings** are added to the embeddings to provide information about the token's position in the sequence. In other words, this allows the model to "know" which token comes first, second, third, and so on in a sentence.
* _Forming input matrices_: The output is a matrix of size `[sequence length × embedding dimension]`, which serves as the initial representation of the text and is passed to the input of the neural network, specifically to the self-attention blocks of the Transformer. Each row of this matrix corresponds to one token, and the vector it contains carries both its semantic meaning (embedding) and information about its position in the text (positional encoding).

Text → Tokens → IDs → Embeddings + Positional Encodings → Model Input

### Attention Mechanism

**The attention mechanism** is a key component of the Transformer architecture that enables the model to account for dependencies between tokens regardless of the _distance_ between them in a sequence. After the input text is converted into a sequence of vectors, this sequence is fed into the central element of the transformer—the **attention block**. 

The attention mechanism is a way for the neural network to determine which parts of the input data to focus on more when processing each element of the sequence. It allows the vectors of individual text fragments to interact with each other, enriching them with information and updating their values based on the surrounding context. This enables the model to effectively capture both local and long-range dependencies between tokens, significantly enhancing its ability to interpret complex text structures. 

In natural language, the meaning of a word or phrase is not determined in isolation; it depends on the context provided by other words and structures. In neural networks, text is encoded through vector representations—embeddings—that numerically reflect the lexical and syntactic properties of tokens. Without a direct attention mechanism (as in the Transformer), contextual information would either be lost over long distances (as in simpler models) or transmitted sequentially, which is less effective for capturing long-range relationships. However, in natural language, the meaning of a word or syntactic construction is dynamic, and its interpretation must adapt to the context. The attention mechanism contextualizes these vector representations, **which means**: 

* Each token assesses its importance relative to other tokens in the sentence.
* During the update process, vector representations are enriched with mutual information, reflecting syntactic dependencies, semantic roles, and pragmatic context.

As a result of this information exchange, the updated vectors begin to encode not only the _meaning of the token itself_ but also its grammatical relationships (syntax), its role in the described situation (semantics), and its overall meaning in context (pragmatics). 

Input Token Vectors (with positions) → Attention Mechanism (Vector Interaction) → Contextualized Token Vectors (Vectors enriched with information about relationships with other tokens)

### Technical Implementation of the Attention Mechanism

The internal workings of the attention mechanism involve several key computational steps and components. For each input vector, three vectors are generated: a Query, a Key, and a Value. Based on their interaction, attention weights are calculated, which are then used to obtain updated, contextualized vector representations. A common approach is to use a Multi-Head Attention architecture for parallel information processing. 

#### Query, Key, and Value

At the core of the attention mechanism's computation is the transformation of each input vector (which is the sum of a token's embedding and its positional encoding) into three distinct vector representations: Query (Q), Key (K), and Value (V). 

Conceptually, these three vectors perform the following roles in the attention mechanism: 

* **Query (Q)**: Represents the current token's vector, which initiates the process of finding relevant information in the sequence. It can be thought of as a "question" or "probe" used to assess the importance of other tokens relative to the current one.
* **Key (K)**: Acts as an identifier or "label" describing an aspect of each token's content. The Query (Q) vector of the current token is compared with all Key (K) vectors in the sequence (including its own) to determine their degree of correspondence or relevance.
* **Value (V)**: Contains the actual information or representation associated with each token that will be passed on. After calculating the attention weights based on the interaction of Queries and Keys, these weights are applied to the Value vectors to form the final weighted representation, which is the output of the attention mechanism for that token.

## Training Large Language Models

LLM training primarily occurs in two stages: 

1. **[Pretraining](https://systems-analysis.ru/eng/Pre-training%5Fof%5Flarge%5Flanguage%5Fmodels "Pre-training of large language models")**: At this stage, the model is trained on large, unlabeled text corpora using self-supervised learning. The task is to predict the next token in a sequence (autoregression) or to restore masked fragments (masked language modeling). Pretraining allows the model to learn broad statistical patterns of language, grammar, facts about the world, and basic forms of reasoning.
2. **[Fine-tuning](https://systems-analysis.ru/eng/Fine-tuning%5F%28deep%5Flearning%29 "Fine-tuning (deep learning)")**: After pretraining, the model is further trained on specialized data to perform specific tasks, such as generating answers, classifying text, or following instructions. Modern approaches include:  
  * Supervised fine-tuning on labeled datasets.
  * [Reinforcement Learning from Human Feedback](https://systems-analysis.ru/eng/Reinforcement%5Flearning%5Ffrom%5Fhuman%5Ffeedback%5F%28RLHF%29 "Reinforcement learning from human feedback (RLHF)") (RLHF) to adjust the model's behavior according to target metrics for quality, safety, and helpfulness.

## Problems and Limitations

Despite impressive progress, modern Large Language Models (LLMs) have a number of problems and limitations: 

**I. Computational and Architectural Limitations** 

1. **High Computational Costs:** Training and operating LLMs require significant computational power, time, and energy, leading to high economic and environmental costs. Additionally, the autoregressive nature of generation (sequential token creation) limits parallelization and slows down inference speed compared to non-autoregressive approaches.
2. **Context Length Limitation:** The Transformer architecture has a quadratic dependency of computational costs and memory requirements on sequence length. This forces a fixed limit ([context window](https://systems-analysis.ru/eng/Context%5Fwindow "Context window")) on the amount of text the model can process at one time, leading to the truncation of long documents and loss of information beyond the window.

**II. Reliability and Accuracy Issues in Generation** 

1. **[Hallucinations](https://systems-analysis.ru/eng/LLM%5Fhallucinations "LLM hallucinations"):** The generation of factually incorrect but plausible-sounding information. This is due to the model's lack of true world understanding, reliance on statistical patterns in data, and inability to verify its generated statements.
2. **Error Propagation:** A process where errors made in the early stages of generation are amplified, leading to an accumulation of inaccuracies in subsequent steps, which degrades the overall quality and coherence of the text.
3. **Limited Compositionality:** Difficulties with tasks requiring multi-step logical reasoning or precise calculations (e.g., multiplying large numbers, solving puzzles). The accuracy of models in such tasks drops sharply as complexity increases due to the autoregressive nature of generation.
4. **Repetition:** A tendency to excessively repeat words or phrases, which reduces the informativeness and readability of the text. This is related to training specifics and decoding algorithms (selection of the next token).
5. **Reversal Curse:** The model's inability to automatically generalize knowledge in the reverse direction: after being trained on the statement "A is B," the model often cannot deduce that "B is A."
6. **Creativity-Accuracy Trade-off:** The need to balance between generating diverse, original responses and maintaining factual accuracy. Enhancing one aspect often negatively affects the other; for example, high creativity can correlate with an increase in hallucinations.

**III. Interaction and Controllability Issues:** 

1. **Poor Controllability:** The difficulty of precisely controlling the style, tone, and content of generated text or ensuring adherence to complex instructions. Controllability is highly dependent on the quality of input instructions ("prompts") and fine-tuning methods.
2. **Sensitivity to Phrasing:** Minor changes in the input prompt can lead to significantly different responses, even if the semantics of the query remain the same. This makes it difficult to obtain stable and predictable results.

**IV. Ethical and Social Aspects:** 

1. **Bias and Fairness Issues:** Models can reproduce and amplify existing social stereotypes, biases, or toxicity present in their training data. Ensuring fairness and safety in models is a complex challenge.
2. **The Alignment Problem (LLM Alignment):** A broader problem that includes the previous point. It is the task of ensuring that a model's behavior aligns with human values, intentions, and ethical norms. It involves combating bias, hallucinations, the generation of harmful content, and improving controllability.
3. **Risks of Malicious Use:** The potential for LLMs to be used for creating and mass-distributing disinformation, phishing, spam, malicious code, or generating convincing fake texts. This poses threats to information and personal security and undermines societal trust.

## External links

* _Large Language Model_ // Wikipedia [(English version)](https://en.wikipedia.org/wiki/Large%5Flanguage%5Fmodel)
* _Grand Modèle de Langage_ // Wikipédia [(French version)](https://fr.wikipedia.org/wiki/Grand%5Fmod%C3%A8le%5Fde%5Flangage)
* _Sprachmodell_ // Wikipedia [(German version)](https://de.wikipedia.org/wiki/Large%5FLanguage%5FModel)
* _Naveed H. et al_. // _A Comprehensive Overview of Large Language Models_ // [arXiv:2307.06435, 2023](https://ar5iv.labs.arxiv.org/html/2307.06435#:~:text=VII%20Challenges%20and%20Future%20Directions)

## Literature

* Vaswani, A. et al. (2017). _Attention Is All You Need_. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
* Devlin, J. et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
* Brown, T. et al. (2020). _Language Models Are Few-Shot Learners_. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165).
* Kaplan, J. et al. (2020). _Scaling Laws for Neural Language Models_. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361).
* Hoffmann, J. et al. (2022). _Training Compute-Optimal Large Language Models_. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556).
* Ouyang, L. et al. (2022). _Training Language Models to Follow Instructions with Human Feedback_. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155).
* Bai, Y. et al. (2022). _Constitutional AI: Harmlessness from AI Feedback_. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073).
* Bubeck, S. et al. (2023). _Sparks of Artificial General Intelligence: Early Experiments with GPT-4_. [arXiv:2303.12712](https://arxiv.org/abs/2303.12712).
* OpenAI. (2023). _GPT-4 Technical Report_. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774).
* Touvron, H. et al. (2024). _The Llama 3 Herd of Models_. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783).

Retrieved from "<https://systems-analysis.ru/eng/index.php?title=Large%5Flanguage%5Fmodel&oldid=2759>"

[Categories](https://systems-analysis.ru/eng/Special:Categories "Special:Categories"): 
* [English](https://systems-analysis.ru/eng/Category:English "Category:English")
* [Large language models](https://systems-analysis.ru/eng/Category:Large%5Flanguage%5Fmodels "Category:Large language models")
* [Machine learning](https://systems-analysis.ru/eng/Category:Machine%5Flearning "Category:Machine learning")
* [Technology](https://systems-analysis.ru/eng/Category:Technology "Category:Technology")

## Navigation menu

### Search

[](https://systems-analysis.ru/eng/Main%5FPage "Visit the main page")

```json
{"@context":"https://schema.org","@type":"WebSite","@id":"https://systems-analysis.ru/#website","url":"https://systems-analysis.ru/","name":"Systems Analysis  EN","inLanguage":"en","publisher":{"@id":"https://systems-analysis.ru/#org"},"isAccessibleForFree":true,"potentialAction":{"@type":"SearchAction","target":{"@type":"EntryPoint","urlTemplate":"https://systems-analysis.ru/wiki/Special:Search?search={search_term_string}"},"query-input":"required name=search_term_string"}}
{"@context":"https://schema.org","@type":"Organization","@id":"https://systems-analysis.ru/#org","name":"Systems Analysis","url":"https://systems-analysis.ru/"}
{"@context":"https://schema.org","@type":"Article","@id":"https://systems-analysis.ru/eng/Large_language_model#article","mainEntityOfPage":"https://systems-analysis.ru/eng/Large_language_model","url":"https://systems-analysis.ru/eng/Large_language_model","inLanguage":"en","headline":"Large language model","name":"Large language model","datePublished":"2026-06-30","dateModified":"2026-06-30","author":{"@id":"https://dmitrishin.github.io/#yuriy-dmitrishin"},"publisher":{"@id":"https://systems-analysis.ru/#org"},"isPartOf":{"@id":"https://systems-analysis.ru/#website"},"identifier":{"@type":"PropertyValue","propertyID":"pageid","value":117}}
```
