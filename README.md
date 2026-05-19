**JCLS 2026 paper "Echoes of Emotion: Linking Narrative and Reader Response of Web Novels in Chinese and English"**

**Code and data for the 2026 JCLS submission on review composition.**

There are 2 folders in **scripts** that include the finetuning and the classification task.

- **finetune**: The finetuning process for our model qidain_webnovel_twitter_xlm_roberta_base_sentiment_multilingual

- **prediction**: The sentiment classification task performed by our finetuned model (mentioned above)

**Figures**: contains tables and figures in the paper

**Data**: contains annotation guidelines, paragraphs and comments with sentiment label.

**synthetic_data**: The synthetic data we generated with LLM for data augmentation for finetuning process.
    
- The notebook llm_synthetic_data_openrouter_api.ipynb is used to generate synthetic data using prompts.
    
- **synthetic_data_eval**: Contains the notebook for synthetic data quality evaluation

**GAM** is the R markdown file for the GAM statistical analysis
