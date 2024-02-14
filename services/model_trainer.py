# https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py
import os

import evaluate
import torch
from datasets import DatasetDict, load_dataset, Audio, IterableDatasetDict
from huggingface_hub import login
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

from services.data_collator import DataCollatorSpeechSeq2SeqWithPadding

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class ModelTrainerService:
    def __init__(self):
        self.hf_token = "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"
        self.num_proc = None
        self.feature_extractor = None
        self.data_collator = None
        self.common_voice = None
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.metric = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    def __compute_metrics(self, pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # we do not want to group tokens when computing the metrics
        label_str = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def __prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_features"] = \
            self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    def login(self):
        login(token=self.hf_token)

    def load(self):
        torch.cuda.empty_cache()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2", token=True)

        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="Lithuanian",
                                                          task="transcribe", token=True)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="Lithuanian",
                                                          task="transcribe", token=True)

        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2", token=True)

        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor,
                                                                  decoder_start_token_id=self.model.config.decoder_start_token_id)

        self.metric = evaluate.load("wer")

    def get_data_source(self):
        self.common_voice = DatasetDict()

        if os.path.exists(os.path.join("./storage", "datasets", "dataset_dict.json")):
            self.common_voice = self.common_voice.load_from_disk(os.path.join("./storage", "datasets"))
        else:
            self.common_voice["train"] = load_dataset("mozilla-foundation/common_voice_16_1", "lt",
                                                      split="train+validation", token=True, trust_remote_code=True)

            self.common_voice["test"] = load_dataset("mozilla-foundation/common_voice_16_1", "lt", split="test",
                                                     token=True, trust_remote_code=True)

            self.common_voice = self.common_voice.cast_column("audio", Audio(sampling_rate=16000))

            self.common_voice = self.common_voice.map(self.__prepare_dataset,
                                                      remove_columns=self.common_voice.column_names["train"],
                                                      num_proc=self.num_proc)

            self.common_voice.save_to_disk(os.path.join("./storage", "datasets"))

    def train(self):
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        # save feature extractor, tokenizer, processor and config

        self.feature_extractor.save_pretrained(os.path.join("./storage", "pretrained"))
        self.tokenizer.save_pretrained(os.path.join("./storage", "pretrained"))
        self.processor.save_pretrained(os.path.join("./storage", "pretrained"))
        self.model.config.save_pretrained(os.path.join("./storage", "pretrained"))
        self.processor.tokenizer.save_pretrained(os.path.join("./storage", "pretrained"))

        # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments

        # UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
        # CUDA out of memory. Tried to allocate 470.00 MiB. GPU 0 has a total capacity of 21.99 GiB of which 73.88 MiB is free. Including non-PyTorch memory, this process has 21.90 GiB memory in use. Of the allocated memory 20.74 GiB is allocated by PyTorch, and 755.52 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
        # `use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`...

        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join("./storage", "output"),
            per_device_train_batch_size=8,  # 8
            per_device_eval_batch_size=8,  # 8
            gradient_accumulation_steps=1,  # 1
            learning_rate=5e-5,  # 5e-5
            warmup_steps=500,  # 0
            num_train_epochs=0.5,  # 3.0
            # The `max_length` to use on each evaluation loop when `predict_with_generate=True`.
            # Will default to the `max_length` value of the model configuration.
            generation_max_length=self.model.config.max_length,
            evaluation_strategy="epoch",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            predict_with_generate=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            save_total_limit=2,
            fp16=True,
            save_strategy="epoch",
            push_to_hub=False,
            report_to=["tensorboard"],
            save_safetensors=False,
        )

        # Initialize a trainer.

        # Multi GPU training in a single process (DataParallel)
        self.model = torch.nn.parallel.DataParallel(self.model, device_ids=[0, 1, 2, 3], dim=0)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.common_voice["train"],
            eval_dataset=self.common_voice["test"],
            data_collator=self.data_collator,
            compute_metrics=self.__compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        # Training

        trainer.train()

        # Saving

        trainer.save_model(os.path.join("./storage", "output"))

        self.processor.tokenizer.save_pretrained(os.path.join("./storage", "output"))
