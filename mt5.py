import warnings
import argparse
import sys
import os
import math
import pandas as pd
import torch
from torch.utils.data import DataLoader
#from transformers import MBartTokenizer, MBartForConditionalGeneration, PreTrainedModel
from transformers import AutoTokenizer,MT5ForConditionalGeneration,PreTrainedModel
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import logging
#from seq2seq_trainer import Seq2SeqTrainer
#from seq2seq_training_args import Seq2SeqTrainingArguments

from transformers.models.bart.modeling_bart import shift_tokens_right
from SARI import SARIsent

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path, sort=False):
        if data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_pickle(data_path)
        # if sort:
        #     self.df = self.df.sample(frac=1, random_state=233).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[[idx]]
        return  { 'comp_txt' : " ".join(row['comp_tokens'].values.tolist()[0]),
                  'simp_txt' : " ".join(row['simp_tokens'].values.tolist()[0])}

    def __len__(self):
        return len(self.df)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,complex_data_path, simple_data_path = None, discard_identical=True):
        self.complex_data = self._load(complex_data_path)
        self.simple_data = None
        if simple_data_path:
            self.simple_data = self._load(simple_data_path)
        if self.simple_data is not None and discard_identical:
            comp_txt,simp_txt=zip(*[(i[0],i[1]) for i in zip(self.complex_data,self.simple_data) if i[0] != i[1]])
            self.complex_data = comp_txt
            self.simple_data = simp_txt

    def _load(self, fname):
        res = []
        for line in open(fname):
            res.append(line.strip().lower())
        return res

    def __getitem__(self, idx):
        return  { 'comp_txt' : self.complex_data[idx].strip().lower(),
                  'simp_txt' : self.simple_data[idx].strip().lower() if self.simple_data else ''
        }

    def __len__(self):
        return len(self.complex_data)

def collate_fn(data):
    """Build mini-batch from a list of examples.
    Note: I don't know why I need this, but if I take it away it doesn't work.
    """
    return data

def convert_to_features(example_batch):
    comp_txt = []
    simp_txt = []
    for ex in example_batch:
        comp_txt.append(ex['comp_txt'])
        simp_txt.append(ex['simp_txt'])

    input_encodings = tokenizer.batch_encode_plus(comp_txt, padding=True,
                                                  max_length=100, truncation=True,
                                                  return_tensors='pt')
    target_encodings = tokenizer.batch_encode_plus(simp_txt, padding=True,
                                                   max_length=100, truncation=True,
                                                   return_tensors='pt')
    labels = target_encodings['input_ids']
    #decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id)
    decoder_input_ids =labels
    #labels[labels[:, :] == model.config.pad_token_id] = -100

    encodings = {
        'input_ids': input_encodings['input_ids'].to(device),
        'attention_mask': input_encodings['attention_mask'].to(device),
        'decoder_input_ids': decoder_input_ids.to(device),
        'labels': labels.to(device)
    }

    return encodings


# because tokenizer.batch_decode fails ...
def decode_ids(ids, tokenizer):
    output = tokenizer.convert_ids_to_tokens(ids)
    assert len(output) > 2, 'Output too short:{}'.format(output)
    # Remove None, BOS, EOS
    end_idx = (output + [None]).index(None)
    output = output[1:end_idx-1]
    return "".join(output).replace('\u0120', ' ')

def generate_outputs(model, tokenizer, data, num_beams):
    outputs = []
    for batch in data:
        comp_sentences = [ex['comp_txt'] for ex in batch]
        inputs = tokenizer(comp_sentences, max_length=1024, padding=True, truncation=True, return_tensors='pt')
        output = model.generate(inputs['input_ids'].to(device), max_length=100) # greedy decoding
        outputs.append(['\n'.join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output])])
    return outputs

def compute_metrics(model, tokenizer, data, num_beams, do_sari):
    loss_list = []
    sari_list = []
    # i = 0
    # print(len(data), flush=True)
    for batch in data:
        # print(i, flush=True)
        # i = i + 1
        inputs = convert_to_features(batch)
        outputs = model(**inputs)
        loss_list.append(outputs["loss"].item() if isinstance(outputs, dict) else outputs[0])
        comp_sentences = [ex['comp_txt'] for ex in batch]
        simp_sentences = [ex['simp_txt'] for ex in batch]
        if not do_sari:
            sari_list.extend([0] * len(comp_sentences))
            continue
        inputs = tokenizer(comp_sentences, max_length=1024, padding=True, truncation=True, return_tensors='pt')
        output = model.generate(inputs['input_ids'].to(device), max_length=100) # greedy decoding
        sys = ['\n'.join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output])]
        sari_list.extend([SARIsent(comp_string, gen_string, [simp_string])
                          for comp_string,simp_string,gen_string in zip(comp_sentences, simp_sentences, sys)])
    #assert len(loss_list) == len(sari_list), f"[E] something went wrong in evaluation! (loss/sari: {len(loss_list)}/{len(sari_list)})"
    return {'loss' : sum(loss_list)/len(loss_list), 'sari':sum(sari_list)/len(sari_list)}

def save_checkpoint(model, tokenizer, optimizer, lr_scheduler, training_arguments):
    '''
    Save checkpoint, which comprises model, tokenizer, optimizer, scheduler and state info.
    '''
    # Save model
    def save_model(model, tokenizer, output_dir, training_args):
        # Create output directory if needed
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(training_args, os.path.join(output_dir, "training_args.bin"))
    output_dir = training_arguments['output_dir']
    save_model(model, tokenizer, output_dir, training_arguments)
    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    with warnings.catch_warnings(record=True) as caught_warnings:
        torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    # reissue_pt_warnings(caught_warnings)

    # # Determine the new best metric / best model checkpoint
    # metric_to_check = "sari"
    # metric_value = metrics['sari']

def eval_and_maybe_save(model, tokenizer, eval_data, optimizer, scheduler, args, best):
    ret = False
    best_str = 'Sari' if args['do_sari'] else 'Loss'
    model.eval()
    logger.info('Starting evaluation')
    metrics = compute_metrics(model, tokenizer, eval_data, args['num_beams'], args['do_sari'])
    cond = best is None or (metrics['sari'] > best if args['do_sari'] else metrics['loss'] < best)
    if cond:
        best = metrics['sari'] if args['do_sari'] else metrics['loss']
        logger.info(f'New best {best_str}, saving model')
        save_checkpoint(model, tokenizer, optimizer, scheduler, args)
        ret = True
    model.train()
    return metrics, ret, best

def train(model, tokenizer, training_data, eval_data, args):
    '''
    model: BART model
    training_data: a Dataset with training data
    eval_data: a Dataset with evaluation data
    arg: arguments. Must have (at least):
        num_train_epochs
        logging_steps
        metric_for_best_model
        learning_rate
        warmup_steps
        weight_decay
        adam_epsilon
        max_grad_norm
        output_dir
        logging_dir
    '''
    num_train_epochs = args['num_train_epochs']
    metric_for_best_model = args['metric_for_best_model'] # 'loss' or 'sari'
    output_dir = args['output_dir']
    logging_dir = args['logging_dir'] if 'logging_dir' in args else output_dir
    max_grad_norm = args['max_grad_norm']

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args['weight_decay']
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args['learning_rate'],
                      betas=(args['adam_beta1'], args['adam_beta2']),
                      eps=args['adam_epsilon'])

    num_update_steps_per_epoch = max(len(training_data), 1)
    check_every = min(args['eval_steps'], num_update_steps_per_epoch)
    log_every = min(args['logging_steps'], check_every)
    max_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)
    num_train_epochs = math.ceil(num_train_epochs)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=max_steps
    )

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # @@ set_seed(args)
    best_eval = None
    best_sari = 0
    best_loss = 0
    best_epoch = 0
    best_global_step = 0
    last_global_step = 0
    model.train()
    for epoch_i in range(num_train_epochs):
        for estep_i, batch in enumerate(training_data):
            inputs = convert_to_features(batch)
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            if global_step % log_every == 0:
                logger.info(f'Epoch: {epoch_i + 1}, Step: {estep_i + 1}, Loss: {tr_loss/global_step:.4f}')
            if global_step % check_every == 0:
                last_eval_step = global_step
                metrics, is_current_best, best_eval = eval_and_maybe_save(model, tokenizer, eval_data, optimizer, scheduler, args, best_eval)
                logger.info(f'Validation: epoch: {epoch_i + 1}, step: {global_step}, dev loss: {metrics["loss"]:.4f}, dev Sari: {metrics["sari"]:.4f}')
                if is_current_best:
                    best_sari = metrics['sari']
                    best_loss = metrics['loss']
                    best_epoch = epoch_i
                    best_global_step = global_step
            # TODO early stopping
            global_step += 1
    if global_step > last_eval_step:
        metrics, is_current_best, best_eval = eval_and_maybe_save(model, tokenizer, eval_data, optimizer, scheduler, args, best_eval)
        logger.info(f'Validation: epoch: {num_train_epochs}, step: {global_step}, dev loss: {metrics["loss"]:.4f}, dev Sari: {metrics["sari"]:.4f}')
        if is_current_best:
            best_sari = metrics['sari']
            best_loss = metrics['loss']
            best_epoch = epoch_i
            best_global_step = global_step
    logger.info(f'Finish training. Best epoch is {best_epoch}, global step:{best_global_step}, loss:{best_loss:.4f}, sari:{best_sari:.4f}')
    return global_step, tr_loss / global_step

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='',
                    help='Training file.')
parser.add_argument('--train_complex_file', type=str, default='',
                    help='Training (text) complex file.')
parser.add_argument('--train_simple_file', type=str, default='',
                    help='Training (text) simple file.')
parser.add_argument('--val_file', type=str, default='',
                    help='Validation file.')
parser.add_argument('--val_complex_file', type=str, default='',
                    help='Validation (text) complex file.')
parser.add_argument('--val_simple_file', type=str, default='',
                    help='Validation (text) simple file.')
parser.add_argument('--test_file', type=str, default='',
                    help='Test file (for predictions).')
parser.add_argument('--predict', action='store_true',
                    help='Predict. Requires --load_model and --test_file.')
parser.add_argument('--output_dir', type=str, default='',
                    help='Directory to store the best model.')
parser.add_argument('--model_name', type=str,default='google/mt5-large',
                    help='Name of BART model.')
parser.add_argument('--load_model', type=str, default='',
                    help='Model to load.')
parser.add_argument('--batch_size', type=int,default=8,
                    help='Number of steps between val check.')
parser.add_argument('--num_epochs', type=int,default=5)
parser.add_argument('--lr', type=float,default=5e-5,
                    help='Learning rate.')
parser.add_argument('--warmup_steps', type=int,default=500,
                    help='Warmup steps before lr scheduler.')
parser.add_argument('--weight_decay', type=float,default=0, # or 0.001?
                    help='Weight decay if we apply some.')
parser.add_argument('--num_beams', type=int,default=4,
                    help='Number of beams for beam search.')
parser.add_argument('--check_every', type=int,default=1000,
                    help='Number of steps between val check.')
parser.add_argument('--log_every', type=int,default=1000,
                    help='Number of steps between logs.')
parser.add_argument('--early_stopping_patience', type=int, default=0,
                    help='Early stop patience.')
parser.add_argument('--no_best_sari', action='store_true',
                    help='Do not use SARI for model selection.')
parser.add_argument('--logfile', type=str, default=None,
                    help='Write to logfile as well as stderr')
parser.add_argument('--nolog', action="store_true", help='Disable logging.')

args = parser.parse_args()

if args.predict:
    assert args.test_file, "Empty --test_file"
    assert args.load_model, "Empty --load_model"
else:
    train_file = []
    val_file = []
    if args.train_file:
        if any([args.train_complex_file, args.train_simple_file]):
            print('Cannot set --train_file and --train_complex_file/train_complex_file at the same time', file=sys.stderr)
            exit(1)
        train_file = [args.train_file]
    else:
        train_file = [args.train_complex_file, args.train_simple_file]
        if not all(train_file):
            print('You need to set --train_file or --train_complex_file/train_complex_file', file=sys.stderr)
            exit(1)
    if args.val_file:
        if any([args.val_complex_file, args.val_simple_file]):
            print('Cannot set --val_train_file and --val_complex_file/val_complex_file at the same time', file=sys.stderr)
            exit(1)
        val_file = [args.val_file]
    else:
        val_file = [args.val_complex_file, args.val_simple_file]
        if not all(val_file):
            print('You need to set --val_file or --val_complex_file/val_complex_file', file=sys.stderr)
            exit(1)
    if not args.output_dir:
            print("Empty --output_dir", file=sys.stderr)
            exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init logger
# logging.set_verbosity_info()
# log = logging.get_logger()
# if args.logfile is not None:
#     import logging
#     log.addHandler(logging.FileHandler(args.logfile))

if args.logfile is not None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s',
                        handlers=[
                            logging.FileHandler(args.logfile),
                            logging.StreamHandler()])
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
if args.nolog:
    logging.disable(sys.maxsize)

logger = logging.getLogger(__name__)
# log argv and git commit
git_commit = 'UNKNOWN'
try:
    import subprocess
    res = subprocess.run('./GIT-VERSION-FILE', stdout=subprocess.PIPE)
    git_commit = res.stdout.decode('utf-8').strip()
except:
    pass
logger.info(f'{git_commit} {" ".join(sys.argv)}')
logger.info(args)

# Check if continuing training from a checkpoint
# if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
#     self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))

model_name_or_path = args.load_model if args.load_model else args.model_name
#tokenizer = MBartTokenizer.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path)
model.to(device)

training_args = {
    'num_train_epochs': args.num_epochs,
    'eval_steps': args.check_every, # steps until evaluating and saving model
    'do_sari' : not args.no_best_sari,
    'logging_steps': args.log_every, # produce log every logging_step steps
    'metric_for_best_model': 'loss' if args.no_best_sari else 'sari',
    'learning_rate': args.lr,
    'warmup_steps':args.warmup_steps,
    'weight_decay':args.weight_decay,
    'adam_beta1':0.9,
    'adam_beta2':0.999,
    'adam_epsilon':1e-8,
    'max_grad_norm':1.0,
    'num_beams':args.num_beams,
    'output_dir':args.output_dir,
    'logging_dir':args.output_dir
}

if args.predict:
    try:
        test_dataset = Dataset(args.test_file)
    except:
        test_dataset = TextDataset(args.test_file, discard_identical = False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False)
    model.eval()
    for batch in test_dataloader:
        inputs = tokenizer(batch['comp_txt'], max_length=1024, padding=True, truncation=True, return_tensors='pt')
        output = model.generate(inputs['input_ids'].to(device), num_beams=4, max_length=100, early_stopping=True)
        print('\n'.join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output]), flush=True)

else:
    if len(train_file) == 1:
        train_dataloader = torch.utils.data.DataLoader(dataset=Dataset(train_file[0]),
                                                       batch_size=args.batch_size,
                                                       collate_fn=collate_fn,
                                                       shuffle=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(dataset=TextDataset(train_file[0], train_file[1]),
                                                       batch_size=args.batch_size,
                                                       collate_fn=collate_fn,
                                                       shuffle=True)
    if len(val_file) == 1:
        val_dataloader = torch.utils.data.DataLoader(dataset=Dataset(val_file[0]),
                                                     #batch_size=args.batch_size,
                                                     batch_size=4,
                                                     collate_fn=collate_fn,
                                                     shuffle=False)
    else:
        val_dataloader = torch.utils.data.DataLoader(dataset=TextDataset(val_file[0], val_file[1]),
                                                     #batch_size=args.batch_size,
                                                     batch_size=4,
                                                     collate_fn=collate_fn,
                                                     shuffle=False)

    train(model, tokenizer, train_dataloader, val_dataloader, training_args)
