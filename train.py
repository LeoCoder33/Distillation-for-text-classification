import argparse
import copy
import os
import logging

import tokenization
import torch
import random
import numpy as np
import pandas as pd
from torch.nn.modules.loss import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.utils.data.dataset import TensorDataset
from tqdm import  tqdm
from progressbar import ProgressBar
from Processor.AGNewsProcessor import AGNewsProcessor
#from models.optimization import BERTAdam
from Processor.DataProcessor import convert_examples_to_features
from models.modeling_single_layer import BertConfig, BertForSequenceClassification
from models.transformers import WarmupLinearSchedule, AdamW

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn)
        # Eval!
        logger.info("********* Running evaluation {} ********".format(prefix))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'albert',
                                                                               'roberta'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("******** Eval results {} ********".format(prefix))
        for key in sorted(result.keys()):
            logger.info(" dev: %s = %s", key, str(result[key]))
    return results

def train(args):
    task_name = args.task_name.lower()

    # Loading BERT model
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    # Identify maximum sequence length matches the allowed bert-maximum sequence length
    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))
    # Identify model
    model = BertForSequenceClassification(bert_config.len(label_list), args.layers, pooling=args.pooling_type)
    # Able to load an pre-trained BERT
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))

    # Configure device and GPU
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    # Distribute model
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    # Processor
    processors = {
        "ag": AGNewsProcessor
    }
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    # Identify train parameter
    train_examples = None
    num_train_steps = None
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size * args.num_train_epochs)  # ???


    # Process train data and test data
    eval_examples = processor.get_dev_examples(args.data_dir, data_num=args.num_test_datas)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, trunc_medium=args.trunc_medium)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)


    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, trunc_medium=args.trunc_medium)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # Start training
    global_step = 0
    epoch = 0
    # Identify optimizer
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_parameters,
                      lr=args.learning_rate,
                      warmup=args.warmup_proportion,
                      t_total=num_train_steps)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # Make a Teacher model
    if args.use_kd:
        kd_model = copy.deepcopy(model)
        kd_loss_fct = MSELoss()
        kd_model.eval()
    for i in range(int(args.num_train_epochs)):
        process_bar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        epoch += 1
        model.train()
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            output = model(input_ids, segment_ids, input_mask, label_ids)
            loss = output[0]
            if args.use_kd:
                input_ids = None
                with torch.no_grad():
                    kd_logits, _ = kd_model(input_ids, segment_ids, input_mask, label_ids)
                # Calculate the MSE loss between teacher and student
                kd_loss = kd_loss_fct(output[1], kd_logits)
                loss += kd_loss * args.kd_coeff
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)

            process_bar(step, {'loss': loss.item()})
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.do_kd:
                    decay = min(args.decay, (1 + global_step) / (10 + global_step))
                    one_minus_decay = 1.0 - decay
                    # Update parameters in Ensemble BERT:
                    with torch.no_grad():
                        parameters = [p for p in model.parameters() if p.requires_grad]
                        for s_param, param in zip(kd_model.parameters(), parameters):
                            s_param.sub_(one_minus_decay * (s_param - param))

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(" ")
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(vocab_path=output_dir)

def main():
    parser = argparse.ArgumentParser()
    #Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--use_kd",
                        default=True,
                        required=True,
                        help="Whether using self-distillation ")
    parser.add_argument("--self_ensemble",
                        default=3,
                        type=int,
                        required=True,
                        help="The number of student model to compose a teacher model")
    parser.add_argument("kd_coeff",
                        type=float,
                        required=True,
                        default=1.0,
                        help="KD loss coefficient")

    # Other parameters
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--trunc_medium',
                        type=int,
                        default=-2,
                        help="choose the trunc ways, -2 means choose the first seq_len tokens, "
                             "-1 means choose the last seq_len tokens, "
                             "0 means choose the first (seq_len // 2) and the last(seq_len // 2). "
                             "other positive numbers k mean the first k tokens "
                             "and the last (seq_len - k) tokens")
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()


    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of 'do_train' or 'do_val' must be True.")

    # Make the output direction
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    if args.do_train:
        train(args)





if __name__ = "__main__":
    main()