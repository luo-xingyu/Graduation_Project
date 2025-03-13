import argparse
import os
import random
from peft import LoraConfig, get_peft_model,TaskType
from matplotlib import pyplot as plt
import numpy as np
from datasets import load_dataset
import evaluate
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments
)
_PARSER = argparse.ArgumentParser('dl detector')
_PARSER.add_argument(
    '-i', '--input', type=str, help='input file path',
    default='hc3'
)
_PARSER.add_argument(
    '-m', '--model-name', type=str, help='model name', default='FacebookAI/roberta-base'
)
_PARSER.add_argument('-b', '--batch-size', type=int, default=8, help='batch size')
_PARSER.add_argument('-g', '--gradient_accumulation_steps', type=int, default=2, help='gradient_accumulation_steps')
_PARSER.add_argument('-e', '--epochs', type=int, default=3, help='epochs')
_PARSER.add_argument('--device', '-d', type=str, default='cuda:0', help='device: cuda:0/cpu')
_PARSER.add_argument('--seed', '-s', type=int, default=42, help='random seed.')
_PARSER.add_argument('--max-length', type=int, default=512, help='max_length')
_PARSER.add_argument("--pair", action="store_true", default=False, help='paired input')
_PARSER.add_argument("--all-train", action="store_true", default=False, help='use all data for training')

_ARGS = _PARSER.parse_args()

def read_train_test(name):
    print(name)
    prefix = 'hc3/'  # path to the csv data from the google drive
    train_name = os.path.join(prefix, name + '_train.csv')
    test_name = os.path.join(prefix, name + '_test.csv')
    dataset = load_dataset('csv', data_files={'train': train_name, 'test': test_name})
    return dataset

def compute_metrics(eval_preds):
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = read_train_test(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess(example):
        return tokenizer(example['question'], example['answer'],max_length=args.max_length,truncation=True).to(args.device)
    remove_columns = ['id', 'question', 'answer', 'source']
    dataset = dataset.map(preprocess,batched=True,remove_columns=remove_columns)
    train_dataset = dataset['train'].rename_column('label', 'labels')
    test_dataset = dataset['test'].rename_column('label', 'labels')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2,id2label={0: "Human", 1: "AI"}).to(args.device) # 标签映射
    output_dir = "./results"  # checkpoint save path
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 序列分类任务
        inference_mode=False,
        r=8,  # 低秩矩阵的维度
        lora_alpha=32,  # 缩放因子
        lora_dropout=0.05,  # 防止过拟合
        target_modules=[  # 针对RoBERTa的特定模块
            "query",
            "key",
            "value",
            "dense"
        ],
        bias="none",  # 不训练偏置参数
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size * 2,
        evaluation_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        fp16=True,  # 启用混合精度训练
        logging_steps=100,
        logging_dir='./logs',
        load_best_model_at_end=True,
        # gradient_checkpointing=True,  # 启用梯度检查点
        # torch_compile=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        peft_model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("./results/AI-detector-1")
    peft_model.save_pretrained("./results/AI-detector-2")
    tokenizer.save_pretrained("./results/AI-detector-2")
    # 获取训练过程中的指标
    print(trainer.state.log_history)
    # 分离训练日志与评估日志
    train_metrics = [log for log in trainer.state.log_history if 'loss' in log]
    eval_metrics = [log for log in trainer.state.log_history if 'eval_loss' in log]
    # 提取训练损失
    train_losses = [log['loss'] for log in train_metrics]
    train_steps = [log['step'] for log in train_metrics]
    # 提取评估指标
    eval_losses = [log['eval_loss'] for log in eval_metrics]
    eval_accuracies = [log['eval_accuracy'] for log in eval_metrics]
    eval_steps = [log['step'] for log in eval_metrics]

    # 绘制图表
    plt.figure(figsize=(12, 5))
    # 绘制损失曲线（训练和评估）
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_losses, 'b-', label="Training Loss")
    plt.plot(eval_steps, eval_losses, 'r--', label="Evaluation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.xticks(np.unique(train_steps))
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    # 绘制评估准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(eval_steps, eval_accuracies, 'g-', marker='o')
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 1.1)
    plt.xticks(np.unique(eval_steps))
    plt.title("Evaluation Accuracy Progress")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(args=_ARGS)
    exit(1)
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