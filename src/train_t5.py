import os
import json
import click
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import (
    AdamW,
    AutoTokenizer,
    T5ForConditionalGeneration
)
from torch import cuda
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

from src.utils.utils import train, encode_data, T5Dataset


@click.command()
@click.option('--train_data_path', type=click.STRING, required=True, default='./data_mapped.csv',
              help='Path to data')
@click.option('--model_name', type=click.STRING, required=True, default='t5-base',
              help='Size of t5 model: t5-small, t5-base, t5-large...')
@click.option('--tokenizer', type=click.STRING, required=True, default='t5-base',
              help='Size of tokenizer: t5-small, t5-base, t5-large...')
@click.option('--epochs', type=click.INT, required=True, default=5,
              help='number of training epochs')
@click.option('--input_data_max_len', type=click.INT, required=True, default=128,
              help='Input data will be tokenized and just this number of tokens considered')
@click.option('--batch_size', type=click.INT, required=True, default=4,
              help='batch size, NB: Consider that the number of batches is then split across GPUs so consider that the number of batches that each epochs train on a single GPU is len(data)//batch_size//n_gpus') # TODO: batchSize with gpus? 
@click.option('--lr', type=click.FLOAT, required=True, default=0.001,
              help='start learning rate')         
@click.option('--output_path', type=click.STRING, required=True, default='./model',
              help='Path & name of folder where model will be saved')
def main(train_data_path, model_name, tokenizer, epochs, input_data_max_len, batch_size, lr, output_path):

    
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model= T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    df=pd.read_csv(os.path.join(train_data_path), sep=',', index_col=0)
    

    data=[]
    max_length=input_data_max_len

    for i,row in df.iterrows():
        data.append(encode_data(tokenizer, {'metadata_str': str(row['Input String']), 'text': row['Text']}, max_length, pad_to_max_length=True, return_tensors="pt"))
    
    dataset= T5Dataset(data, tokenizer, max_length)
    
    train_loader=DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    print('Initiating Fine-Tuning for the model on our dataset')
    losses=[]
    for epoch in range(epochs):
        epoch_loss=train(tokenizer, model, device, train_loader, optimizer)
        losses.append(epoch_loss.detach().cpu().numpy())
        print(f'Epoch: {epoch}, Loss:  {epoch_loss}')
    print('End Fine-Tuning')

    plt.plot(range(epochs), losses)
    plt.title('loss along epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('trainig.png')


    # if not os.path.exists(os.path.join(output_path, model)):
    #     print(f'Created the folder: {os.path.join(output_path, model)}')
    #     os.mkdir(os.path.join(output_path, model))
    model.save_pretrained(os.path.join(output_path, model_name))

if __name__ == "__main__":
    main()