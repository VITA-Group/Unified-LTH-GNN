import os
import argparse
from pathlib import Path
from v2GNNAccel.utils.getting_metrics import get_embedding_dimension, get_number_classes
from v2GNNAccel.utils.reading_graphs import read_gml_graph_to_pyg

os.environ["PYTHONWARNINGS"] = "ignore"


def main(args):
    lt_modes = args.lt_modes
    dataset = args.dataset
    dataset_path = args.dataset_path
    num_layers = args.num_layers
    lr = args.lr
    weight_decay = args.weight_decay
    pruning_percent_wei = args.pruning_percent_wei
    pruning_percent_adj = args.pruning_percent_adj
    total_epoch = args.total_epoch
    hidden_dimensions = args.hidden_dimensions
    models = args.models
    gml_dataset_path = args.gml_dataset_path
    csv_path = args.csv_path

    graph_names = set()
    for root, directories, files in os.walk(dataset_path):
        for filename in files:
            graph_names.add(filename.split(".")[1])

    csv_filename = f"{dataset_path.split('/')[-1]}-models({','.join(models)})-hidden({','.join([str(hid) for hid in hidden_dimensions])})-num_layers({','.join([str(nl) for nl in num_layers])}).csv"

    for lt_mode in lt_modes:
        for n_Layers in num_layers:
            for hidden in hidden_dimensions:
                for dataset in graph_names:
                    dataset_path = f"{args.dataset_path}/{dataset}"
                    graph_id = dataset
                    gml_source = f"{gml_dataset_path}/{dataset}"
                    pyg_data = read_gml_graph_to_pyg(gml_source)
                    d = get_embedding_dimension(pyg_data)
                    c = get_number_classes(pyg_data)
                    embedding_dim = f"{d} {' '.join([str(hidden) for _ in range(n_Layers)])} {c}"  # embedding_dim should be as large as the number of hidden layers we want in the middle and the value is the layer dim
                    command = f"""{args.python} {lt_mode} --dataset {dataset} \
                        --dataset_path {dataset_path} --embedding-dim {embedding_dim} --lr {lr} \
                        --weight-decay {weight_decay} --pruning_percent_wei {pruning_percent_wei} --pruning_percent_adj {pruning_percent_adj} --total_epoch {total_epoch} --graph_id {graph_id} \
                        --csv_filename '{csv_path+csv_filename}'
                        """
                    print(command)
                    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments script")
    parser.add_argument(
        "--lt_modes",
        nargs="+",
        type=str,
        default=[
            "/home/polp/puigde/v2GNNAccel/Unified-LTH-GNN/NodeClassification/main_pruning_random.py"
        ],
        help="List of LT modes",
    )
    parser.add_argument("--dataset", type=str, default="grph_6", help="Dataset name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/polp/puigde/v2GNNAccel/datasets/minisample/lt",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--num_layers",
        nargs="+",
        type=int,
        default=[2, 4, 6],
        help="List of numbers of layers",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument(
        "--pruning_percent_wei",
        type=float,
        default=0.2,
        help="Pruning percentage for weights",
    )
    parser.add_argument(
        "--pruning_percent_adj",
        type=float,
        default=0.05,
        help="Pruning percentage for adjacency",
    )
    parser.add_argument(
        "--total_epoch", type=int, default=20, help="Total number of epochs"
    )
    parser.add_argument(
        "--hidden_dimensions",
        nargs="+",
        type=int,
        default=[16, 32, 64],
        help="List of hidden dimensions",
    )
    parser.add_argument(
        "--models", nargs="+", type=str, default=["gcn"], help="Available models"
    )
    parser.add_argument(
        "--gml_dataset_path",
        type=str,
        default="/home/polp/puigde/v2GNNAccel/datasets/minisample/",
        help="Path to GML datasets",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/polp/puigde/v2GNNAccel/experiments/lt/",
        help="Path for saving CSV files",
    )

    parser.add_argument(
        "--python",
        type=str,
        default="python",
        help="Path to the python executable",
    )

    args = parser.parse_args()
    main(args)
