# Neural Network Scaling Insights

Training large neural networks (in 2024, meaning more than hundred billions of params) is not trivial; people say scaling is all you need, but are you sure you can replicate GPT-4 models when you have sufficient amount of compute? This material is a collection of insights about training "large" neural networks, aka how to scale your models. **Please recommend me more materials that I am missing**! Also, if you notice any mistakes, please let me know.

Acknowledgement: I would recommend reading this [insane blog](https://cloneofsimo.notion.site/What-to-do-to-scale-up-09e469d7c3444d6a90305397c38a46f5) from [@cloneofsimo](https://x.com/cloneofsimo), which is a great resource for scaling up neural networks and motivated me a lot to study this topic.

## How to set your batchsize?

- 18', [An Empirical Model of Large-Batch Training](https://arxiv.org/pdf/1812.06162): Use gradient noise scale to determine the critical batch size (for optimal compute/time tradeoffs) for model training (larger batchsize = smaller loss improvement from single step.) Discuss the trade-off between the time and the computational cost; in general, small batch size is better for the compute-optimality. Also discussed changes of the critical batch size during training, dynamic batch size may help?
- More readings (not carefully read yet)
  - Awesome blog post: 24', [How to Scale Hyperparameters as Batch Size Increases](https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/): Using SDE to understand the impact of batch size on gradient noise scale.
  - 19', [Measuring the Effects of Data Parallelism on Neural Network Training](https://www.jmlr.org/papers/v20/18-789.html)
  - 19', [Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model](https://proceedings.neurips.cc/paper/2019/hash/e0eacd983971634327ae1819ea8b6214-Abstract.html)

## How to stabilize your training?

WIP, but thinking of reading these papers more carefully:

- 23', [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322)

## How to initialize your model?

WIP. Now studying mup.

## Which data should I use? -- on the focus of language modeling

### Data filtering and deduplication

Lots of papers are discussing the importance of high-quality data, but the notion of "high-quality" can be very subjective. Sharing some notable empirical results:

- 24', [Dolma](https://arxiv.org/abs/2402.00159): open resources for data construction, including the fine-grained details and codes for data curation.
- 24', [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) and 24', [FineWeb-Edu](https://huggingface.co/blog/smollm): Filtering and deduplication are the keys for high-quality data, textbook quality is all you need? Trained an "educational quality classifier" to filter the data.

### Data mixtures

- 23', [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429): use learnability ([rho-loss](https://proceedings.mlr.press/v162/mindermann22a.html)) to upweight the data that is more learnable, worth learning, and not yet learnt. Also works in [vision models](https://arxiv.org/abs/2312.05328).
  - One caveat is that it uses a proxy model to find the weights; empirical results show that it works well, but there are several evidences ([LESS](https://arxiv.org/abs/2402.04333), [Data Mixing Law](https://arxiv.org/abs/2403.16952)) that the proxy model is not perfect.


## Loss power laws

One important case of the physics of deep learning. The validation loss follows the power laws (*log (loss) = - a x + b*) when scaling the number of model params and the number of data points (and also the number of compute). This leads to defining the compute-optimality when training the models -- given the amount of compute, which model size and data size is the optimal to reach the lowest validation loss?

- 20', [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) and 20', [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/pdf/2010.14701): redicting the test loss of the LM and the other autoregressive models when the input is the number of model params and the number of data points. Surprising to see that the loss follows the power laws ACROSS ALL DOMAINS, including text-to-image, image-to-text, and videos. More takeaways:
  - Model shape (width/depth ratio) was not crucial factor,
  - Larger models can achieve the same test loss with less amount of data,
  - Compute-optimal point != model convergence.

- 21', [Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers](https://arxiv.org/pdf/2109.10686): downstream task performance can't be predicted by the perplexity; model shape is important for the downstream task. Here, suggests deepening the model instead of widening the model. 

- 21', [Scaling Laws for Transfer](https://arxiv.org/pdf/2102.01293): "Effective data transferred" follows the power law in the data regime, which means that pre-training enables models to achieve lower test loss on target task (fine-tuning setups. But should consider the "ossification" phenomena, where the model shows worse test loss when the model is pre-trained; this is shown in the low-parameter regime.

- 22', [Training compute-optimal large language models](https://arxiv.org/abs/2203.15556): aka chinchilla paper; obtained different fitted power laws from the scaling law before. Key takeaway is that the models are under-trained, so need to scale the data more than before.
  - Why different power laws? -- cosine learning rate schedule should consider the number of total steps, which is not considered in the previous scaling law paper.

- 23', [Scaling Data-Constrained Language Models](https://arxiv.org/pdf/2305.16264) and 24', [Scaling Laws for Data Filtering— Data Curation cannot be Compute Agnostic](https://arxiv.org/pdf/2404.07177#page=6.06): Previous scaling law only considered when the data is unlimited (so training for one epoch), what if the data is limited? First paper says that similar to Chinchilla-optimal, allocate more compute on data until it is not repeating too much; 4 epochs of training is okay for the experiments. Note that Chinchilla paper also didn't consider repeating data. Second paper assumes the heterogeneity of the utility of the data, showing the quality-quantity tradeoff of the data which should be considered when finding the compute-optimal point.

- 23', [Scaling Laws for Reward Model Overoptimization](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf) and 24', [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/pdf/2406.02900): discussing the trade-off between reward and the kl-divergence -- higher KL divergence between the initial policy and the trained policy means the higher reward score. Practically meaningful to find the optimal point of the RLHF training.

### Subtopic: how to explain the power laws?

WIP. But some papers to read:

- 21', [Learning Curve Theory](https://arxiv.org/pdf/2102.04074)
- 22', [A Solvable Model of Neural Scaling Laws](https://arxiv.org/pdf/2210.16859)
- 23', [Explaining neural scaling laws](https://www.pnas.org/doi/epdf/10.1073/pnas.2311878121)

## Finding compute multipliers

To scale your model better, you should optimize MFU (Model FLOPS Utilization)! If you can make the training faster for 30% with some tricks (of course you can bring some algorithmic advances), it means that you have 30% more GPUs. This is called the compute multiplier, mentioned by Dario Amodei (Anthropic). So, what are the things?

- Using [Triton](https://openai.com/index/triton/) to make your training code faster (or even faster with custom CUDA kernels),
- Writing good distributed training codes (learning [DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/master) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) would be a great starter), like removing unnecessary overheads in the codes,
- Reducing the bottleneck of the models, like attention mechanism -- Flash-attention series ([1](https://arxiv.org/abs/2205.14135), [2](https://arxiv.org/abs/2307.08691), [3](https://arxiv.org/abs/2407.08608)) and [PagedAttention](https://arxiv.org/abs/2309.06180) (in vllM; though this is now mostly for the inference time).

Or, you can even look for the model architecture innovations, such as using state-space models (like [Mamba](https://arxiv.org/pdf/2312.00752)) or other non-autoregressive models -- note that there are no clear evidences that these models will work on par with Transformer-based models.

## How to setup the learning rate schedule?

WIP. But some papers to read:

- 24', [Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler](https://arxiv.org/abs/2408.13359)
- WSD scheduler (from vision transformers, minicpm, )
- Scheduler-free training

## Optimizers

WIP. But some papers to read:

- 18', [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/pdf/1802.09568)

## Other topics to read

Controlled experiments (using small amount of GPUs) to understand the language models better:

- 23', [Physics of language models: Part 3.1, knowledge storage and extraction](https://arxiv.org/pdf/2309.14316)
- 23', [Physics of language models: Part 3.2, knowledge manipulation](https://arxiv.org/pdf/2309.14402)
- 24', [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405)
- 24', [Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process](https://arxiv.org/abs/2407.20311)
- 24', [Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://www.arxiv.org/abs/2408.16293)
