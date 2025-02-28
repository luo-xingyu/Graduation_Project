"""
pip install datasets evaluate scikit-learn torch==1.12.1 transformers
"""

import argparse
import os
import random

_PARSER = argparse.ArgumentParser('dl detector')
_PARSER.add_argument(
    '-i', '--input', type=str, help='input file path',
    default='en'
)
_PARSER.add_argument(
    '-m', '--model-name', type=str, help='model name', default='roberta-base'
)
_PARSER.add_argument('-b', '--batch-size', type=int, default=8, help='batch size')
_PARSER.add_argument('-e', '--epochs', type=int, default=2, help='batch size')
_PARSER.add_argument('--cuda', '-c', type=str, default='0', help='gpu ids, like: 1,2,3')
_PARSER.add_argument('--seed', '-s', type=int, default=42, help='random seed.')
_PARSER.add_argument('--max-length', type=int, default=512, help='max_length')
_PARSER.add_argument("--pair", action="store_true", default=False, help='paired input')
_PARSER.add_argument("--all-train", action="store_true", default=False, help='use all data for training')


_ARGS = _PARSER.parse_args()

# if len(_ARGS.cuda) > 1:
#     os.environ['TOKENIZERS_PARALLELISM'] = 'false'
#     os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# os.environ["OMP_NUM_THREADS"] = '8'
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # if cuda >= 10.2
# os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda


def main(args: argparse.Namespace):
    import numpy as np
    from datasets import Dataset, concatenate_datasets
    import evaluate
    import pandas as pd
    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def read_train_test(name):
        print(name)
        prefix = 'hc3/'  # path to the csv data from the google drive
        train_df = pd.read_csv(os.path.join(prefix, name + '_train.csv'))
        test_df = pd.read_csv(os.path.join(prefix, name + '_test.csv'))
        len(train_df)
        len(test_df)
        print(train_df.head())
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        print(train_dataset)
        print(test_dataset)
        return train_dataset, test_dataset

    if 'mix' in args.input:
        data = [read_train_test(args.input.replace('mix', m)) for m in ('text', 'sent')]
        train_dataset = concatenate_datasets([data[0][0], data[1][0]])
        test_dataset = concatenate_datasets([data[0][1], data[1][1]])
    else:
        train_dataset, test_dataset = read_train_test(args.input)

    if args.all_train:
        train_dataset = concatenate_datasets([train_dataset, test_dataset])
        print("Using all data for training..")
        print(train_dataset)
        test_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    kwargs = dict(max_length=args.max_length, truncation=True)
    if args.pair:
        def tokenize_fn(example):
            return tokenizer(example['question'], example['answer'], **kwargs)
    else:
        def tokenize_fn(example):
            return tokenizer(example['answer'], **kwargs)

    print('Tokenizing and mapping...')
    train_dataset = train_dataset.map(tokenize_fn)
    if test_dataset is not None:
        test_dataset = test_dataset.map(tokenize_fn)

    # remove unused columns
    names = ['id', 'question', 'answer', 'source']
    tokenized_train_dataset = train_dataset.remove_columns(names)
    if test_dataset is not None:
        tokenized_test_dataset = test_dataset.remove_columns(names)
    else:
        tokenized_test_dataset = None
    print(tokenized_train_dataset)

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    output_dir = "./results"  # checkpoint save path
    if args.pair:
        output_dir += '-pair'
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='no' if test_dataset is None else 'steps',
        eval_steps=2000 if 'sent' in args.input else 500,
        save_strategy='epoch',
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # 获取训练过程中的指标
    train_losses = trainer.state.log_history["train_loss"]
    eval_losses = trainer.state.log_history["eval_loss"]
    eval_accuracies = trainer.state.log_history["eval_accuracy"]

    # 绘制图表
    epochs = list(range(1, training_args.num_train_epochs + 1))
    plt.figure(figsize=(10, 5))

    # 绘制训练和评估损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, eval_losses, label="Evaluation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Losses")
    plt.legend()

    # 绘制评估准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, eval_accuracies, label="Evaluation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy")
    plt.legend()

    plt.show()

if __name__ == '__main__':
    import os
    # import json
    # import matplotlib.pyplot as plt

    # # 从train_state.json文件中读取数据
    # with open('./models/roberta/trainer_state.json', 'r') as file:
    #     train_state = json.load(file)

    # # 提取损失值和准确率值
    # loss_values = [log['loss'] for log in train_state['log_history'] if 'loss' in log]
    # accuracy_values = [log['eval_accuracy'] for log in train_state['log_history'] if 'eval_accuracy' in log]

    # # 创建损失曲线图
    # plt.figure(figsize=(10, 5))
    # plt.plot(loss_values, label='Loss', color='blue')
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # 创建准确率曲线图
    # plt.figure(figsize=(10, 5))
    # plt.plot(accuracy_values, label='Accuracy', color='green')
    # plt.xlabel('Step')
    # plt.ylabel('Accuracy')
    # plt.title('Evaluation Accuracy')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    import os
    import json
    import matplotlib.pyplot as plt

    # 从第一个模型的train_state.json文件中读取数据
    with open('./models/1.json', 'r') as file:
        train_state_1 = json.load(file)

    # 提取第一个模型的损失值和准确率值
    loss_values_1 = [log['loss'] for log in train_state_1['log_history'] if 'loss' in log]
    accuracy_values_1 = [log['eval_accuracy'] for log in train_state_1['log_history'] if 'eval_accuracy' in log]
    learning_rate_values_1 = [log['learning_rate'] for log in train_state_1['log_history'] if 'learning_rate' in log]

    # 从第二个模型的train_state.json文件中读取数据
    with open('./models/deberta-v3-base/trainer_state.json', 'r') as file:
        train_state_2 = json.load(file)

    # 提取第二个模型的损失值和准确率值
    loss_values_2 = [log['loss'] for log in train_state_2['log_history'] if 'loss' in log]
    accuracy_values_2 = [log['eval_accuracy'] for log in train_state_2['log_history'] if 'eval_accuracy' in log]
    learning_rate_values_2 = [log['learning_rate'] for log in train_state_2['log_history'] if 'learning_rate' in log]

    # 创建损失曲线图
    plt.figure(figsize=(10, 5))
    plt.plot([i * 500 for i in range(len(loss_values_1))], loss_values_1, label='BERT Loss', color='blue')
    plt.plot([i * 500 for i in range(len(loss_values_2))], loss_values_2, label='RoBERTa Loss', color='red')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 创建准确率曲线图
    plt.figure(figsize=(10, 5))
    plt.plot([i * 500 for i in range(len(accuracy_values_1))], accuracy_values_1, label='BERT Accuracy', color='green')
    plt.plot([i * 500 for i in range(len(accuracy_values_2))], accuracy_values_2, label='RoBERTa Accuracy', color='orange')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 创建学习率曲线图
    # plt.figure(figsize=(10, 5))
    # plt.plot([i * 500 for i in range(len(learning_rate_values_1))], learning_rate_values_1, label='Model 1 Learning Rate', color='purple')
    # plt.plot([i * 500 for i in range(len(learning_rate_values_2))], learning_rate_values_2, label='Model 2 Learning Rate', color='cyan')
    # plt.xlabel('Step')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # main(_ARGS)