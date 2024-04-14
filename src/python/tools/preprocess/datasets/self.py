from pathlib import Path
from marius.tools.preprocess.dataset import NodeClassificationDataset
from marius.tools.preprocess.utils import download_url, extract_file
import numpy as np
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.datasets.ogb_helpers import remap_ogbn
import torch
import numpy as np
from omegaconf import OmegaConf
from offgs.dataset import OffgsDataset
import argparse
import sys
sys.path.append('/nvme2n1/baseline_marius/src/python/tools/preprocess/datasets')
from load_graph import *


class SELF(NodeClassificationDataset):

    def __init__(self, output_directory: Path):

        super().__init__(output_directory)

        self.dataset_name = "ogbn_papers100M"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"

    def download(self, overwrite=False):

        self.input_edge_list_file = self.output_directory / Path("data.npz")    # key: edge_index
        self.input_node_feature_file = self.output_directory / Path("data.npz") # key: node_feat
        self.input_node_label_file = self.output_directory / Path("node-label.npz")
        self.input_train_nodes_file = self.output_directory / Path("train.csv")
        self.input_valid_nodes_file = self.output_directory / Path("valid.csv")
        self.input_test_nodes_file = self.output_directory / Path("test.csv")

        download = False
        if not self.input_edge_list_file.exists():
            download = True
        if not self.input_node_feature_file.exists():
            download = True
        if not self.input_node_label_file.exists():
            download = True
        if not self.input_train_nodes_file.exists():
            download = True
        if not self.input_valid_nodes_file.exists():
            download = True
        if not self.input_test_nodes_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)

            (self.output_directory / Path("papers100M-bin/raw/data.npz")).rename(self.input_node_feature_file)
            (self.output_directory / Path("papers100M-bin/raw/node-label.npz")).rename(self.input_node_label_file)

            for file in (self.output_directory / Path("papers100M-bin/split/time")).iterdir():
                extract_file(file)

            for file in (self.output_directory / Path("papers100M-bin/split/time")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="ogbn-products")
        parser.add_argument("--store-path", type=str, default="/nvme1n1/offgs_dataset")
        parser.add_argument("--path", type=str, default="/efs/rjliu/dataset/igb_full")
        parser.add_argument("--dataset_size", type=str, default="full")
        parser.add_argument("--num_classes", type=int, default=19)
        parser.add_argument("--in_memory", type=int, default=0)
        parser.add_argument("--synthetic", type=int, default=0)
        args = parser.parse_args()
        print(args)
        args.dataset = self.custom_name


        label_offset = 0
        if args.dataset.startswith("ogbn"):
            dataset = load_ogb(args.dataset, "/efs/rjliu/dataset")
        elif args.dataset.startswith("igb"):
            dataset = load_igb(args)
        elif args.dataset == "mag240m":
            dataset = load_mag240m("/efs/rjliu/dataset/mag240m", only_graph=False)
            label_offset = dataset[-1]
            dataset = dataset[:-1]
        elif args.dataset == "friendster":
            dataset = load_friendster("/efs/rjliu/dataset/friendster", 128, 20)
        else:
            raise NotImplementedError

        print(dataset[0])
        print("Preprocessing SELF dataset")
        ## print self.custom_dataset
        print('self.custom_dataset',self.custom_name)
        dataset_path=f'/nvme1n1/offgs_dataset/{self.custom_name}-offgs'
        import os
        graph_pth=os.path.join(dataset_path,'graph.pth')
        graph_load=torch.load(graph_pth)
        if self.custom_name=='ogbn-arxiv':
            graph_load=graph_load.remove_self_loop()
        edges_np = np.stack((graph_load.edges()[0].numpy(), graph_load.edges()[1].numpy()), axis=1)
        spllit_idx_load=torch.load(os.path.join(dataset_path,'split_idx.pth'))
        train_nodes=spllit_idx_load['train'].numpy().astype(np.int32)
        if self.custom_name=='igb-full':
            train_nodes=torch.load('/efs/rjliu/dataset/igb_full/train_idx_0.1.pt').numpy().astype(np.int32)
        valid_nodes=spllit_idx_load['valid'].numpy().astype(np.int32)
        test_nodes=spllit_idx_load['test'].numpy().astype(np.int32)
        ## TODO check here
        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=edges_np,
            num_partitions=num_partitions,
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            format="NUMPY",
            known_node_ids=[train_nodes, valid_nodes, test_nodes,np.arange(graph_load.num_nodes(), dtype=np.int32)]
        )
        
        # exit(1)

        dataset_stats = converter.convert()

        features = dataset[1].numpy().astype(np.float32)
        
        labels = np.load(os.path.join(dataset_path,'labels.npy')).astype(np.int32)
        if self.custom_name=='mag240m':
            zeros_to_add = np.zeros(label_offset, dtype=np.int32)
            labels = np.concatenate([zeros_to_add, labels])

        if remap_ids:
            print('remap_ids')
            node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
            print('node_mapping')
            train_nodes, valid_nodes, test_nodes, features, labels = remap_ogbn(node_mapping, train_nodes, valid_nodes, test_nodes, features, labels)

        with open(self.train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))
        with open(self.valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))
        with open(self.test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))
        with open(self.node_features_file, "wb") as f:
            f.write(bytes(features))
        with open(self.node_labels_file, "wb") as f:
            f.write(bytes(labels))
        # exit(1)

        # update dataset yaml
        dataset_stats.num_train = train_nodes.shape[0]
        dataset_stats.num_valid = valid_nodes.shape[0]
        dataset_stats.num_test = test_nodes.shape[0]
        dataset_stats.feature_dim = features.shape[1]
        dataset_stats.num_classes = dataset[3]

        dataset_stats.num_nodes = graph_load.num_nodes()


        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)
        print('dataset_stats all')
        return dataset_stats

