# Football Transfer Market News Generation

In this repository you can find the slides and demo for **Football Transfer Market News Generation** session, presented (in Italian) at Global AI Developer Days, Turin Conference, on October 28th, 2022.

Abstract:

In this session I'll explore how to automatically generate football transfer-market news using [T5 model](https://arxiv.org/pdf/1910.10683.pdf) for conditional generation: after a brief overview about NLP (Natural Language Processing) and NLG (Natural Language Generation), I will fine-tune a pre-trained model on some football transfer-market news tweets.

Speakers:

- [Federico Barbiero](https://www.linkedin.com/in/federico-barbiero-87374b171) (Deltatre)

---

## Setup local environment

Hardware requirements:

- NVIDIA GPU  

Software requirements:

- Windows 10 21H2 or Windows 11
- NVIDIA drivers, CUDA 11.6 or higher, with cuDNN properly configured for Windows (follow [instructions here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html))
- Visual Studio Code
- Python 3.9.x

To setup a local copy, just clone the repository.  
You can find the training notebook in the `notebooks` folder, while training scripts and scoring demo app are in `src` folder. Slides can be found in the `docs` folder.

## Create the environment

To run notebook and scripts, create a Python environment, and install the required packages.

```ps
python -m pip install -U pip
python -m venv venv
pip install requirements.txt
```

## Create the Dataset

To run the application, you have to create a compatible dataset, to do so you need to:

- create a Twitter Developer Account (see the [relative doc](doc\twitter_API.md)), scrape some news account, and run the `twitter_api.py` file.
- create a model for Named Entity Recognition. __I can not divulgate the model I used or the data I trained with because it is a Deltatre private model based on proprietary data__.


## Model Fine Tuning Demo

To run the demo code, you first need to train and create a proper model: use 
`src\train_t5.py` with the proper parameters (such as model size, batch size, max_length ...). The code will download the pretrained model from [HuggingFace](https://huggingface.co/docs/transformers/model_doc/t5) models zoo and then fine tune it on the Twitter data.

Once you run this, you can try the model both using `notebooks\models_combo.ipynb` (it is a notebook that runs in cascade both the NER model and T5) or use the Streamlit webpage to see the quick `demo web_page.py` (run it by launching on Powershell `streamlit run web_page.py`)


## References and other useful links

- <https://arxiv.org/pdf/1910.10683.pdf>
- <https://huggingface.co/docs/transformers/model_doc/t5>
- <https://developer.twitter.com/en/portal/dashboard>
- <https://www.tweepy.org/>
- <https://ai.googleblog.com/2021/01/totto-controlled-table-to-text.html>
- <https://paperswithcode.com/sota/data-to-text-generation-on-totto>
- <https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html>
- <https://github.com/google-research/bert>
- <https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.>


## License

---

Copyright (C) 2022 Deltatre.  
Licensed under [MIT license](./LICENSE).