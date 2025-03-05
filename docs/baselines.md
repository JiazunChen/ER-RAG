# CRAG Baselines

For the CRAG benchmark, we provide participants with two baseline models to help get started. Detailed implementations of these baseline models are accessible through the links provided below. Participants are encouraged to use these as a starting point for the competition.

Please note that these baselines are **NOT** tuned for performance or efficiency, and are provided as is for demonstration.


## Available Baseline Models:

1. [**Vanilla Llama 2 Model**](../models/vanilla_llama_baseline.py): For an implementation guide and further details, refer to the Vanilla Llama 2 model documentation [here](../models/vanilla_llama_baseline.py).

2. [**RAG Baseline Model**](../models/rag_llm_model.py): For an implementation guide and further details, refer to the RAG Baseline model documentation [here](../models/rag_llm_model.py).

## Preparing Your Submission:

Before you can submit your solutions using these baselines, it is necessary to download the model weights and incorporate them into this repository. To do this, follow the step-by-step instructions outlined in the document: [download_baseline_model_weights.md](download_baseline_model_weights.md). 

Additionally, ensure that your configurations in [user_config.py](../models/user_config.py) correctly reference the model class you intend to use for your submission.

These steps are crucial for a successful submission. Make sure to follow them carefully. Good luck!
