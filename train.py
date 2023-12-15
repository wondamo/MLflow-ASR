from datasets import load_dataset
from transformers import Wav2Vec2Processor, TrainingArguments, Trainer, Wav2Vec2ForCTC
from dataclasses import dataclass
from typing import Any, Union, List, Dict, Optional

import torch
import evaluate
import numpy as np
import mlflow
import os


os.environ["MLFLOW_EXPERIMENT_NAME"] = "speechnotetaking-asr"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "s3://mlflow-dba"
os.environ["MLFLOW_TRACKING_URI"] = "postgresql+psycopg2://postgres:zTx2e3pn79YMkDKtzttf@database-1.cvsqlcx445xv.us-east-1.rds.amazonaws.com:5432/mlflow_db"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"

model_checkpoint = "facebook/wav2vec2-base"

tedlium = load_dataset("LIUM/tedlium", "release1", split="train[:100]")

processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

tedlium = tedlium.map(prepare_dataset, remove_columns=tedlium.column_names["train"], num_proc=4)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

wer = evaluate.load('wer')

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model = Wav2Vec2ForCTC.from_pretrained(
    model_checkpoint,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)


training_args = TrainingArguments(
    output_dir="speechnotetaking_asr_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tedlium["train"],
    eval_dataset=tedlium["test"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

mlflow.end_run()