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

if args.seed is not None:
    base_model_save_path = f"checkpoints_{args.seed}/{args.model}"
else:
    base_model_save_path = f"checkpoints/{args.model}"

with open(os.path.join(base_model_save_path, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = [
    "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
]

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

control_dataset = "ImageNet"
negation_accuracies = {}
mode = args.finetuning_mode

for dataset in eval_datasets:
    task_vector = None
    pretrained_checkpoint = None

    if mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        if not (os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint)):
            print(f"Warning: Missing checkpoints for {dataset}. Skipping.")
            continue
        task_vector = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    elif mode == "linear_ortho":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_ortho_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_ortho_finetuned.pt"
        if not (os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint)):
            print(f"Warning: Missing checkpoints for {dataset}. Skipping.")
            continue
        task_vector = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    elif mode == "standard_ortho":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/standard_ortho_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/standard_ortho_finetuned.pt"
        if not (os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint)):
            print(f"Warning: Missing checkpoints for {dataset}. Skipping.")
            continue
        task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    elif mode in ("linear-2", "linear-2_ortho"):
        prefix = mode + "_"
        pretrained_checkpoint = f"{args.save}/{dataset}Val/{prefix}zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/{prefix}finetuned.pt"
        if not (os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint)):
            print(f"Warning: Missing checkpoints for {dataset}. Skipping.")
            continue
        task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    else:  # standard
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        if not (os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint)):
            print(f"Warning: Missing checkpoints for {dataset}. Skipping.")
            continue
        task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    if not os.path.exists(pretrained_checkpoint):
        print(f"Error: Base pretrained checkpoint not found at {pretrained_checkpoint}. Skipping {dataset}.")
        continue

    task_vector = -task_vector

    args.eval_datasets = [dataset + "Val"]
    args.control_dataset = control_dataset + "Val"
    val_metrics = evaluate_task_vector(
        task_vector,
        pretrained_checkpoint,
        args,
        posthoc_linearization=False,
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}Val:top1",
        minimize=True,
        control_metric=f"{control_dataset}Val:top1",
        control_metric_threshold=args.control_threshold * pretrained_accuracies[control_dataset + "Val"],
    )

    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        optimal_coef,
        posthoc_linearization=False,
    )

    print("=" * 100)
    print(f"Results for dataset: {dataset}")
    print(f"Optimal Coefficient: {optimal_coef}")
    print(f"Test accuracy: {test_metrics.get(f'{dataset}:top1', 'N/A')}")
    print(f"Control accuracy on {control_dataset}: {test_metrics.get(f'{control_dataset}:top1', 'N/A')}")

    negation_accuracies[dataset] = {
        "test": test_metrics.get(f"{dataset}:top1"),
        "test_control": test_metrics.get(f"{control_dataset}:top1"),
        "val": val_metrics,
        "optimal_coef": optimal_coef,
    }

save_name_map = {
    "standard": "negations.json",
    "standard_ortho": "standard_ortho_negations.json",
    "linear": "linear_negations.json",
    "linear_ortho": "linear_ortho_negations.json",
    "linear-2": "linear-2_negations.json",
    "linear-2_ortho": "linear-2_ortho_negations.json",
}
save_file = os.path.join(args.save, save_name_map[mode])
with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)
print(f"Negation results saved to {save_file}")
