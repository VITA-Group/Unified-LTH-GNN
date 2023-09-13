# python main_pruning_random.py --dataset grph_6 --dataset_path /home/polp/puigde/gnn_accel/datasets/minisample/lt --embedding-dim 413 512 19 --lr 0.01 --weight-decay 5e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 200

from gnn_accel.utils.getting_metrics import get_embedding_dimension, get_number_classes
from gnn_accel.utils.reading_graphs import read_gml_graph_to_pyg
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# models = ['gcn', 'gin', 'agnn']
# # model = 'gcn' # gcn, gin, agnn
# hidden = [16, 32, 64]
# filepaths = []

# d is the input embedding dimension and c is the number of classes
# c is the number of output classes

def main():
	lt_modes = ["main_pruning_random.py"]
	dataset = "grph_6"
	dataset_path = "/home/polp/puigde/gnn_accel/datasets/minisample/lt"
	num_layers = [2, 4, 6]
	lr = f"{0.01}"
	weight_decay = f"{5e-4}"
	pruning_percent_wei = f"{0.2}"
	pruning_percent_adj = f"{0.05}"
	total_epoch = f"{20}"
	hidden_dimensions = [16, 32, 64]
	models = ['gcn']

	gml_dataset_path = "/home/polp/puigde/gnn_accel/datasets/minisample/"
	dataset_path = f"/home/polp/puigde/gnn_accel/datasets/minisample/lt/"
	graph_names = set()
	for root, directories, files in os.walk(dataset_path):
		for filename in files:
			graph_names.add(filename.split(".")[1])
	csv_filename =  f"{dataset_path.split('/')[-1]}-models({','.join(models)})-hidden({','.join([str(hid) for hid in hidden_dimensions])})-num_layers({','.join([str(nl) for nl in num_layers])}).csv"
	csv_path = "/home/polp/puigde/gnn_accel/experiments/lt/"

	for lt_mode in lt_modes:
		for n_Layers in num_layers:
			for hidden in hidden_dimensions:
				for dataset in graph_names:
					dataset_path = f"/home/polp/puigde/gnn_accel/datasets/minisample/lt/{dataset}/"
					gml_source = gml_dataset_path+dataset
					pyg_data = read_gml_graph_to_pyg(gml_source)
					d = get_embedding_dimension(pyg_data)
					c = get_number_classes(pyg_data)
					embedding_dim = f"{d} {' '.join([str(hidden) for _ in range(n_Layers)])} {c}" # embedding_dim should be as large as the number of hidden layers we want in the middle and the value is the layer dim
					command = f"""python {lt_mode} --dataset {dataset} \
						--dataset_path {dataset_path} --embedding-dim {embedding_dim} --lr {lr} \
						--weight-decay {weight_decay} --pruning_percent_wei {pruning_percent_wei} --pruning_percent_adj {pruning_percent_adj} --total_epoch {total_epoch} \
						--csv_filename '{csv_path+csv_filename}'
						"""
					print(command)
					os.system(command)

if __name__ == "__main__":
	main()