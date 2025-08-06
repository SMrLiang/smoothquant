import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
import tqdm
import torch.nn.functional as F
from datasets import load_dataset
import argparse
import random
MAX_SEQ_LEN = 128
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="act_scales/llama-2-7b.pt",
)
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--w_bit", type=int, default=8)
parser.add_argument("--a_bit", type=int, default=8)


args = parser.parse_args()
alpha = args.alpha
model_path = args.model_path
act_scales_path = args.act_scales_path
n_samples = args.n_samples
w_bit = args.w_bit
a_bit = args.a_bit

def get_epoch_batches(tokenizer, device, dataset, micro_batch_size=4, shuffle=True, drop_last=False, seed=42):
    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)    # 每个 epoch 随机打乱
    # num = 2048
    # for start_idx in range(0, num, micro_batch_size):
    for start_idx in range(0, len(indices), micro_batch_size):
        batch_indices = indices[start_idx:start_idx + micro_batch_size]
        samples = [dataset[i] for i in batch_indices]

        texts = []
        labels = []

        for sample in samples:
            goal = sample["goal"]
            sol1 = sample["sol1"]
            sol2 = sample["sol2"]
            correct_label = int(sample["label"])  # 0 or 1

            # 构造两个输入，一个对应 sol1，一个对应 sol2
            texts.append(f"{goal}\nOption: {sol1}")
            texts.append(f"{goal}\nOption: {sol2}")

            # 正确答案在哪个文本上就标为 1，其它为 0
            labels.append(1 if correct_label == 0 else 0)
            labels.append(1 if correct_label == 1 else 0)

        inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(labels).to(device)
        # 返回一个 dict，带有 labels
        # inputs["labels"] = torch.tensor(labels).to(device)
        # inputs = tokenizer("DeepSpeed is great!", return_tensors="pt", padding=True)
        yield inputs

def get_seq_logprobs(logits, input_ids, attention_mask):
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    # batch_size, seq_len
    mask = attention_mask[:, 1:] if attention_mask is not None else torch.ones_like(targets)
    # logits [batch, seq_len, voc_size]
    log_prob = F.log_softmax(logits, dim=-1)
    # 按照id选择对应token的probs -> batch, seq_len
    token_logprobs = torch.gather(log_prob, 2, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * mask).sum(dim=1)

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = dataset

        # self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total_sample = 0
        quant_correct = 0 
        for step, batch in tqdm.tqdm(enumerate(get_epoch_batches(tokenizer, self.device, self.dataset, micro_batch_size=1024)), desc="testing", leave=False):
            labels = batch.get("labels", None)
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            nlls = []
            quant_pred = []
            batch_size = labels.shape[0] // 2
            gt_labels = torch.zeros(batch_size, dtype=torch.long, device=labels.device)
            
            for i in range(batch_size):
                if labels[2 * i] == 1:
                    gt_labels[i] = 0  # 正确答案是第一个输入
                else:
                    gt_labels[i] = 1  # 正确答案是第二个输入
            with torch.no_grad():
                lm_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            assert labels is not None
            
            q_seq_logprobs = get_seq_logprobs(lm_logits, input_ids, attention_mask)
            for i in range(batch_size):
                # sol1 vs sol2

                if q_seq_logprobs[2 * i] < q_seq_logprobs[2 * i + 1]:
                    quant_pred[i] = 1
                else:
                    quant_pred[i] = 0

            quant_correct += (quant_pred == gt_labels).sum().item()
            total_sample += gt_labels.numel()

        # torch.stack在新维度拼接，cat是在已有维度上。对于0维元素cat不了
        # 用sum(nlls)也一样
        print("total samples: ", total_sample)
        return quant_correct / total_sample


tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("piqa", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples)

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)

if args.smooth:
    act_scales = torch.load(act_scales_path)
    smooth_lm(model, act_scales, alpha)
if args.quantize:
    import inspect
    # print(inspect.getfile(quantize_model))
    model = quantize_model(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=True,
        w_bit=w_bit,
        a_bit=a_bit,
    )

acc = evaluator.evaluate(model)
print(f"Accuracy: {acc}")
