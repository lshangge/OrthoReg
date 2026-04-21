import json
import os

from utils import find_optimal_coef

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.attention_only_finetune import AttentionOnlyFinetuneEncoder

args = parse_arguments()

if 'ortho' in args.finetuning_mode:
    args.save = f"checkpoints_{args.seed}/{args.finetuning_mode}_{args.lr}_lambda{args.ortho_lambda}_{args.model}"
else:
    if args.seed is not None:
        args.save = f"checkpoints_{args.seed}/{args.finetuning_mode}_{args.lr}_{args.model}"
    else:
        args.save = f"checkpoints/{args.finetuning_mode}_{args.lr}_{args.model}"

print("*" * 100)
mode_labels = {
    "standard": "Evaluating non-linear FT models.",
    "standard_ortho": "Evaluating standard FT models with orthogonality regularization.",
    "linear": "Evaluating linear FT models.",
    "linear_ortho": "Evaluating linear FT models with orthogonality regularization.",
    "linear-2": "Evaluating Attention-Only Finetune models.",
    "linear-2_ortho": "Evaluating Attention-Only Finetune models with orthogonality regularization.",
}
ft_accuracies_name_map = {
    "standard": "ft_accuracies.json",
    "standard_ortho": "standard_ortho_ft_accuracies.json",
    "linear": "linear_ft_accuracies.json",
    "linear_ortho": "linear_ortho_ft_accuracies.json",
    "linear-2": "linear-2_ft_accuracies.json",
    "linear-2_ortho": "linear-2_ortho_ft_accuracies.json",
}
print(mode_labels.get(args.finetuning_mode, f"Evaluating {args.finetuning_mode} models."))
print("*" * 100)

ft_accuracies_path = os.path.join(args.save, ft_accuracies_name_map[args.finetuning_mode])
with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

if args.seed is not None:
    base_model_save_path = f"checkpoints_{args.seed}/{args.model}"
else:
    base_model_save_path = f"checkpoints/{args.model}"

with open(os.path.join(base_model_save_path, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = [
    "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN", "SUN397",
]

task_vectors = []
mode = args.finetuning_mode

for dataset in eval_datasets:
    if mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        task_vectors.append(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint))

    elif mode == "linear_ortho":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_ortho_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_ortho_finetuned.pt"
        task_vectors.append(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint))

    elif mode == "standard_ortho":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/standard_ortho_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/standard_ortho_finetuned.pt"
        task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

    elif mode in ("linear-2", "linear-2_ortho"):
        prefix = mode + "_"
        pretrained_checkpoint = f"{args.save}/{dataset}Val/{prefix}zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/{prefix}finetuned.pt"
        if not (os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint)):
            print(f"Warning: Missing checkpoints for {dataset}. Skipping.")
            continue
        task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

    else:  # standard
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

if not task_vectors:
    print("No task vectors were created. Exiting.")
    exit()

task_vector = sum(task_vectors)

# Determine the base pretrained checkpoint
mode_prefix_map = {
    "standard": "",
    "standard_ortho": "standard_ortho_",
    "linear": "linear_",
    "linear_ortho": "linear_ortho_",
    "linear-2": "linear-2_",
    "linear-2_ortho": "linear-2_ortho_",
}
mode_prefix = mode_prefix_map[mode]
pretrained_checkpoint = f"{args.save}/{eval_datasets[0]}Val/{mode_prefix}zeroshot.pt"

if not os.path.exists(pretrained_checkpoint):
    print(f"Error: Base pretrained checkpoint not found at {pretrained_checkpoint}")
    exit()

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    args,
    posthoc_linearization=False,
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)

args.eval_datasets = [dataset for dataset in eval_datasets]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
    posthoc_linearization=False,
)

print("=" * 100)
print(f"Optimal Coefficient: {optimal_coef}")
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics, "optimal_coef": optimal_coef}

save_name_map = {
    "standard": "additions.json",
    "standard_ortho": "standard_ortho_additions.json",
    "linear": "linear_additions.json",
    "linear_ortho": "linear_ortho_additions.json",
    "linear-2": "linear-2_additions.json",
    "linear-2_ortho": "linear-2_ortho_additions.json",
}
save_file = os.path.join(args.save, save_name_map[mode])
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
print(f"Addition results saved to {save_file}")
