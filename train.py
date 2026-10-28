"""Optional SQuAD 2.0 fine-tuning script.
Run on a GPU machine/Colab rather than inside the demo app.
"""
import json
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer,
    DefaultDataCollator
)

from config import MODEL_DIR, MODEL_NAME, TRAIN_SAMPLES, VALIDATION_SAMPLES, EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_LENGTH, DOC_STRIDE


def prepare_features(examples, tokenizer):
    questions = [q.strip() for q in examples['question']]
    inputs = tokenizer(
        questions, examples['context'], max_length=MAX_LENGTH, truncation='only_second',
        stride=DOC_STRIDE, return_overflowing_tokens=True, return_offsets_mapping=True,
        padding='max_length'
    )
    sample_map = inputs.pop('overflow_to_sample_mapping')
    offsets = inputs.pop('offset_mapping')
    start_positions, end_positions = [], []
    for i, offsets_i in enumerate(offsets):
        sample_idx = sample_map[i]
        answer = examples['answers'][sample_idx]
        if not answer['text']:
            start_positions.append(0); end_positions.append(0); continue
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        sequence_ids = inputs.sequence_ids(i)
        context_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
        context_end = len(sequence_ids) - 1 - next(j for j, s in enumerate(reversed(sequence_ids)) if s == 1)
        if offsets_i[context_start][0] > start_char or offsets_i[context_end][1] < end_char:
            start_positions.append(0); end_positions.append(0); continue
        while context_start < len(offsets_i) and offsets_i[context_start][0] <= start_char:
            context_start += 1
        start_positions.append(context_start - 1)
        while offsets_i[context_end][1] >= end_char:
            context_end -= 1
        end_positions.append(context_end + 1)
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs


def main():
    dataset = load_dataset('squad_v2')
    train = dataset['train'].select(range(min(TRAIN_SAMPLES, len(dataset['train']))))
    valid = dataset['validation'].select(range(min(VALIDATION_SAMPLES, len(dataset['validation']))))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = train.map(lambda x: prepare_features(x, tokenizer), batched=True, remove_columns=train.column_names)
    tokenized_valid = valid.map(lambda x: prepare_features(x, tokenizer), batched=True, remove_columns=valid.column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / 'checkpoints'),
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        report_to='none',
        fp16=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_train, eval_dataset=tokenized_valid, tokenizer=tokenizer, data_collator=DefaultDataCollator())
    trainer.train()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / 'threshold.json').write_text(json.dumps({'threshold': 0.0}, indent=2))
    print(f'Model saved to {MODEL_DIR}')


if __name__ == '__main__':
    main()
