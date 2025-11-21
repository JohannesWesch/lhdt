import torch
from src.data.basic import BasicDataModule
from src.HDT import HDTTokenizer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from src.utils.data_collators import DataCollatorForMaskedLanguageModeling
from datasets import load_dataset, concatenate_datasets, load_from_disk, IterableDataset, interleave_datasets
from torch.utils.data.dataloader import DataLoader
from src.utils import module_to_dict
from transformers import AutoTokenizer
import configs as CONFIG
from src.utils import *


class MLMDataModule(BasicDataModule):
    _name_ = "mlm"

    def __init__(self):
        super().__init__()

    def prepare_data(self) -> None:
        if not self.prepared:
            # When using streaming, we just use the pre-trained tokenizer
            use_streaming = getattr(CONFIG.data_config, 'use_streaming', True)
            
            if use_streaming:
                # Use pre-trained tokenizer for streaming
                if CONFIG.data_config.tok_name not in ["BPE", "Unigram", "WordLevel", "WordPiece", "WordPieceBERT", "SentencePieceUnigram", "SentencePieceBPE"]:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        CONFIG.cfg_data.tok_name, 
                        cls_token="<cls>", 
                        bos_token="<s>",
                        model_max_length=CONFIG.model_config.max_encoder_position_embeddings
                    )
                    # Add mask_token if not present (e.g., T5 doesn't have one by default)
                    if self.tokenizer.mask_token is None:
                        self.tokenizer.add_special_tokens({'mask_token': '<mask>'})
                    
                    # Add special tokens if they are defined (e.g., for hierarchical models)
                    if hasattr(self, "special_tokens") and self.special_tokens:
                        special_tokens_to_add = []
                        for tok_name, tok in self.special_tokens.items():
                            if tok_name == "additional_special_tokens" and isinstance(tok, list):
                                self.tokenizer.add_special_tokens({tok_name: tok}, replace_additional_special_tokens=False)
                            elif tok_name in self.tokenizer.special_tokens_map.keys():
                                self.tokenizer.add_special_tokens({tok_name: tok})
                            else:
                                special_tokens_to_add.append(tok)
                        if special_tokens_to_add:
                            self.tokenizer.add_tokens(special_tokens_to_add)
                    self.tokenizer.save_pretrained(CONFIG.save_dir)
            else:
                # Download datasets and train tokenizer
                corpus_list = []
                for cfg_dict in CONFIG.cfg_data.ds_info:
                    raw_dataset = load_dataset(**cfg_dict, cache_dir=CONFIG.cache_dir)
                    corpus_list.append(raw_dataset)
                raw_data = self._concatenate_datasets(corpus_list)
                self.tokenizer = self._get_tokenizer(raw_data)
                self.tokenizer.save_pretrained(CONFIG.save_dir)
            self.prepared = True

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.save_dir)
        self.data_collator = DataCollatorForMaskedLanguageModeling(self.tokenizer, CONFIG.cfg_data.mlm_probability, input_max_length=CONFIG.cfg_data.model_max_length, hierarchical=CONFIG.hierarchical)
        if stage == "fit":
            corpus_list = []
            use_streaming = getattr(CONFIG.data_config, 'use_streaming', True)
            
            for cfg_dict in CONFIG.cfg_data.ds_info:
                if use_streaming:
                    raw_dataset = load_dataset(**cfg_dict, streaming=True)
                else:
                    raw_dataset = load_dataset(**cfg_dict, cache_dir=CONFIG.cache_dir)
                corpus_list.append(raw_dataset)
            
            # Use interleave_datasets for streaming, concatenate_datasets for non-streaming
            if use_streaming and isinstance(corpus_list[0], IterableDataset):
                self.data_train = interleave_datasets(corpus_list, seed=CONFIG.cfg_exps.seed)
                
                # For multi-GPU training with streaming, shard the dataset per rank
                if self.multi_gpu and torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    rank = torch.distributed.get_rank()
                    self.data_train = self.data_train.shard(num_shards=world_size, index=rank)
            else:
                self.data_train = self._concatenate_datasets(corpus_list)
                self._log_tokenization(self.data_train)
            # self.data_train = preprocess(self.data_train, self.tokenizer, self.cfg_data)
        elif stage == "test":
            ## Validation set always use AG_news
            test_dataset = load_dataset("ag_news", name="default", split="train", num_proc=CONFIG.cfg_data.num_proc,
                                        cache_dir=CONFIG.cache_dir)
            self.data_test = test_dataset.remove_columns([col for col in test_dataset.column_names if col != "text"])
            # self.data_test = preprocess(test_dataset, self.tokenizer, self.cfg_data)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.data_train[-1000:], batch_size=CONFIG.exps_config.batch_size,
                          num_workers=CONFIG.cfg_data.num_proc, collate_fn=self.data_collator)


class HierarchicalMLMDataModule(MLMDataModule):
    _name_ = "hierarchical_mlm"
    special_tokens = dict(cls_token="<cls>", sec_token="<sec>", doc_token="<doc>")

    def _log_tokenization(self, train_dataset):
        hdt_tokenizer = HDTTokenizer(self.tokenizer, CONFIG.cfg_data.model_max_length)
        # 4) Log overviews so we always know what's going on with weird tokenization tricks
        random_sentence_idx = torch.randint(0, len(train_dataset), (1,)).item()
        input_data = train_dataset[random_sentence_idx]["text"]
            # .squeeze()  # squeeze because hf has leading dim
        dataset_size = len(train_dataset)

        log.info(
            f"Random sentence with seq_length {CONFIG.cfg_data.model_max_length} from dataset of size {dataset_size:,}: ...")
        log.info(input_data)
        log.info("... is tokenized into ...")
        tokenized_doc = hdt_tokenizer(input_data)["input_ids"]
        log.info(" ".join(self.tokenizer.decode(t) for t in tokenized_doc))


