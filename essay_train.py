import argparse
import os
import random
from peft import LoraConfig, get_peft_model,TaskType
from matplotlib import pyplot as plt
import numpy as np
from datasets import load_dataset,Dataset,ClassLabel
import evaluate
import pandas as pd
import torch,re,itertools
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments,set_seed,
)
from sklearn.metrics import (  # Import various metrics from scikit-learn
    accuracy_score,  # For calculating accuracy
    roc_auc_score,  # For ROC AUC score
    confusion_matrix,  # For confusion matrix
    classification_report,  # For classification report
    f1_score  # For F1 score
)
_PARSER = argparse.ArgumentParser('dl detector')
_PARSER.add_argument(
    '-i', '--input', type=str, help='input file path',
    default='data'
)
_PARSER.add_argument(
    '-m', '--model-name', type=str, help='model name', default='distilbert/distilbert-base-uncased-finetuned-sst-2-english' # distilbert/distilbert-base-cased distilbert/distilroberta-base distilbert/distilbert-base-uncased-finetuned-sst-2-english albert/albert-base-v2 albert/albert-large-v1
)
_PARSER.add_argument('-b', '--batch-size', type=int, default=8, help='batch size')
_PARSER.add_argument('-g', '--gradient_accumulation_steps', type=int, default=2, help='gradient_accumulation_steps')
_PARSER.add_argument('-e', '--epochs', type=int, default=2, help='epochs')
_PARSER.add_argument('--device', '-d', type=str, default='cuda:0', help='device: cuda:0/cpu')
_PARSER.add_argument('--seed', '-s', type=int, default=42, help='random seed.')
_PARSER.add_argument('--max-length', type=int, default=512, help='max_length')
_PARSER.add_argument("--train_fraction", default=0.9, help='paired input')
_PARSER.add_argument("--learning_rate", default=5e-7, help='paired input')
_PARSER.add_argument("--warmup_steps", default=50, help='paired input')
_PARSER.add_argument("--weight_decay", default=0.02, help='paired input')
_PARSER.add_argument("--all-train", action="store_true", default=False, help='use all data for training')

_ARGS = _PARSER.parse_args()
labels_list = ['Human', 'AI']
def read_train_test(name):
    df0 = pd.read_csv("./data/Training_Essay_Data.csv", encoding='latin-1')
    df0 = df0.rename(columns={'generated': 'labels'})  # Rename the columns to standard ones
    print(df0.shape, df0.columns)
    df1 = pd.read_csv("./data/final_train.csv")
    df1 = df1.rename(columns={'label': 'labels'})  # Rename the columns to standard ones
    print(df1.shape, df1.columns)
    df2 = pd.read_csv("./data/final_test.csv")
    df2 = df2.rename(columns={'label': 'labels'})  # Rename the columns to standard ones
    print(df2.shape, df2.columns)
    df = pd.concat([df0, df1, df2], axis=0)
    item0 = df.shape[0]  # Store the initial number of items in the DataFrame
    df = df.drop_duplicates()  # Remove duplicate rows from the DataFrame
    item1 = df.shape[0]  # Store the number of items in the DataFrame after removing duplicates
    print(f"There are {item0 - item1} duplicates found in the dataset")  # Print the number of duplicates removed
    df = df[['labels', 'text']]  # Select only the 'label' and 'text' columns
    df = df[~df['text'].isnull()]  # Remove rows where 'text' is null
    df = df[~df['labels'].isnull()]  # Remove rows where 'label' is null
    def clean_text(text, stem=True):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        return text
    def change_label(x):
        if x:
            return 'AI'
        else:
            return 'Human'
    df['labels'] = df['labels'].apply(change_label)
    df['text'] = df['text'].apply(clean_text)
    print(df.sample(5))  # Display a random sample of 5 rows from the DataFrame
    # Initialize empty dictionaries to map labels to IDs and vice versa
    label2id, id2label = dict(), dict()
    # Iterate over the unique labels and assign each label an ID, and vice versa
    for i, label in enumerate(labels_list):
        label2id[label] = i  # Map the label to its corresponding ID
        id2label[i] = label  # Map the ID to its corresponding label
    print(label2id, id2label)
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(labels_list)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df['labels'])
    # Create a dictionary mapping each class to its respective class weight.
    class_weights = dict(zip(classes, weights))
    ordered_weigths = [class_weights[x] for x in id2label.values()]
    print(ordered_weigths)
    dataset = Dataset.from_pandas(df)
    return dataset,label2id, id2label,ordered_weigths

def compute_metrics(eval_preds):
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


def main(args: argparse.Namespace):
    set_seed(42)
    dataset,label2id, id2label,ordered_weigths = read_train_test(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,use_fast=True, low_cpu_mem_usage=False)
    # Creating classlabels to match labels to IDs
    ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)
    def preprocess(example):
        example['labels'] = ClassLabels.str2int(example['labels'])
        return tokenizer(example['text'],max_length=args.max_length,truncation=True).to(args.device)
    remove_columns = ['text']
    dataset = dataset.map(preprocess,batched=True,remove_columns=remove_columns)
    dataset = dataset.cast_column('labels', ClassLabels)

    # Splitting the dataset into training and testing sets using the predefined train/test split ratio.
    dataset = dataset.train_test_split(test_size=1 - args.train_fraction, shuffle=True, stratify_by_column="labels")
    #df_train = dataset['train'].select(range(1000))
    #df_test = dataset['test'].select(range(200))
    df_train = dataset['train']
    df_test = dataset['test']
    print(df_train.shape, df_test.shape)
    #print(tokenizer.decode(df_train[0]['input_ids']))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels_list),
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id = label2id).to(args.device) # 标签映射
    output_dir = "./results"  # checkpoint save path
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 序列分类任务
        inference_mode=False,
        r=8,  # 低秩矩阵的维度
        lora_alpha=32,  # 缩放因子
        lora_dropout=0.05,  # 防止过拟合
        target_modules=["query", "key", "value", "ffn_output"],
        bias="none",  # 不训练偏置参数
    )
    '''target_modules=["query","key","value","dense"] #针对RoBERT
            ["query", "key", "value", "ffn_output"] #针对alBERT
            ["q_lin", "k_lin", "v_lin", "out_lin"], #针对BERT特定模块'''
    #peft_model = get_peft_model(model, peft_config)
    #peft_model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size * 8,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy='steps',
        eval_steps=2000,
        save_strategy='steps',
        save_steps=2000,
        fp16=True,  # 启用混合精度训练
        logging_steps=2000,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        # gradient_checkpointing=True,  # 启用梯度检查点
        # torch_compile=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Print the computed class weights to the console.
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs,num_items_in_batch=None,return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has labels with different weights)
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(ordered_weigths, device=model.device).float())
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    trainer = WeightedTrainer(
        model,
        training_args,
        train_dataset=df_train,
        eval_dataset=df_test,
        data_collator=data_collator,
        processing_class=tokenizer, #旧版本:tokenizer=tokenizer
        compute_metrics=compute_metrics,
    )
    #print(trainer.evaluate())
    trainer.train()
    print(trainer.evaluate())
    # Use the trained 'trainer' to make predictions on the 'df_test'.
    outputs = trainer.predict(df_test)
    # Print the metrics obtained from the prediction outputs.
    print("output metrics: ",outputs.metrics)
    model_path = "./results/AI-detector-" + args.model_name.replace("/", "-")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    # 获取训练过程中的指标
    print("train log: ",trainer.state.log_history)
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
    plt.figure(figsize=(16, 5))
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
    plt.savefig(model_path+"/loss and accuracy.png")
    plt.show()

    y_true = outputs.label_ids

    # Predict the labels by selecting the class with the highest probability
    y_pred = outputs.predictions.argmax(1)

    # Define a function to plot a confusion matrix
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8), is_norm=True):
        """
        This function plots a confusion matrix.

        Parameters:
            cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
            classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
            title (str): Title for the plot.
            cmap (matplotlib colormap): Colormap for the plot.
        """
        # Create a figure with a specified size
        plt.figure(figsize=figsize)

        # Display the confusion matrix as an image with a colormap
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        # Define tick marks and labels for the classes on the axes
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        if is_norm:
            fmt = '.3f'
        else:
            fmt = '.0f'
        # Add text annotations to the plot indicating the values in the cells
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        # Label the axes
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Ensure the plot layout is tight
        plt.tight_layout()
        plt.savefig(model_path + "/matrix.png")
        plt.show()

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Display accuracy and F1 score
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Get the confusion matrix if there are a relatively small number of labels
    if len(labels_list) <= 120:
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize='true')

        # Plot the confusion matrix using the defined function
        plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

    # Finally, display classification report
    print()
    print("Classification report:")
    print()
    print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

if __name__ == '__main__':
    main(args=_ARGS)