from argparse import ArgumentParser
import gc
import time
import warnings

from pytorchcrf import CRF as CRF_2
import torch
from TorchCRF import CRF as CRF_3
from torchcrf import CRF as CRF_4

from linear_crf import LinearCRF as CRF_1


warnings.filterwarnings("ignore", category=FutureWarning)

parser = ArgumentParser(description="Benchmark the different packages")
parser.add_argument("-n", "--num-batches", help="Number of batches", type=int, default=1000)
parser.add_argument("-b", "--batch-size", help="Batch size", type=int, default=100)
parser.add_argument("-l", "--seq-length", help="Sequence length", type=int, default=100)
parser.add_argument("-t", "--num-tags", help="Number of tags", type=int, default=10)
parser.add_argument("--num-tries", help="Number of tries", type=int, default=5)
parser.add_argument("-s", "--seed", help="Random seed", type=int, default=22)
args = parser.parse_args()


print("Evaluating models with the following parameters:")
print(f"\tnumber of batches: {args.num_batches}")
print(f"\tbatch size: {args.batch_size}")
print(f"\tsequence length: {args.seq_length}")
print(f"\tnumber of tags: {args.num_tags}")
print(f"\trandom seed: {args.seed}")
print()

torch.manual_seed(args.seed)
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print(local_msg)
    print(f"Total execution time = {end_time - start_time:.3f} sec")
    print(f"Max memory used by tensors = {torch.cuda.max_memory_allocated() / (1024**2)} MB")


mask = torch.ones(args.batch_size, args.seq_length, dtype=torch.uint8, device="cuda")

for try_id in range(1, args.num_tries + 1):
    print(f"TRY {try_id}")
    print("~~~~~\n")
    data = [
        torch.randn(args.batch_size, args.seq_length, args.num_tags)
        for _ in range(args.num_batches)
    ]
    targets = [
        torch.randint(args.num_tags, (args.batch_size, args.seq_length))
        for _ in range(args.num_batches)
    ]

    model_1 = CRF_1(args.num_tags, batch_first=True)
    model_2 = CRF_2(args.num_tags, batch_first=True)
    model_3 = CRF_3(args.num_tags)
    model_4 = CRF_4(args.num_tags, batch_first=True)

    with torch.no_grad():
        model_2.start_transitions.copy_(model_1.starts)
        model_2.transitions.copy_(model_1.transitions)
        model_2.end_transitions.copy_(model_1.ends)
        model_3.start_trans.copy_(model_1.starts)
        model_3.trans_matrix.copy_(model_1.transitions)
        model_3.end_trans.copy_(model_1.ends)
        model_4.start_transitions.copy_(model_1.starts)
        model_4.transitions.copy_(model_1.transitions)
        model_4.end_transitions.copy_(model_1.ends)

    for model, label in zip(
        [model_1, model_2, model_3, model_4], ["linear_crf", "pytorchcrf", "TorchCRF", "torchcrf"]
    ):
        model.to("cuda")
        print(f"benchmarking {label} package")
        print("~" * len(f"benchmarking {label} package"))
        start_timer()
        for x, y in zip(data, targets):
            loss = model(x.to("cuda"), y.to("cuda"), mask=mask).mean()
            loss.backward()
        end_timer_and_print("Forward + backward pass:")
        print()
        start_timer()
        with torch.no_grad():
            for x in data:
                if label == "TorchCRF":
                    model.viterbi_decode(x.to("cuda"), mask=mask)
                else:
                    model.decode(x.to("cuda"), mask=mask)
        end_timer_and_print("Viterbi decode:")
        print()
