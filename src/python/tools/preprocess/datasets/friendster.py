

from pathlib import Path
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file, strip_header
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter


class Friendster(LinkPredictionDataset):

    def __init__(self, output_directory: Path):
        super().__init__(output_directory)

        self.dataset_name = "twitter"
        self.dataset_url = "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"

    def download(self, overwrite=False):

        self.input_edges = self.output_directory / Path("com-friendster.ungraph.txt")

        if not self.input_edges.exists():
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=True)
            strip_header(self.input_edges, num_lines=4)

    def preprocess(self,
                   num_partitions=1,
                   remap_ids=True,
                   splits=None,
                   sequential_train_nodes=False,
                   generate_random_features=False,
                   feature_dim=32,
                   num_classes=50,
                   node_splits=[.1, .05, .05]):

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_edges,
            delim="\t",
            columns=[0, 1],
            header_length=0,
            num_partitions=num_partitions,
            splits=splits,
            remap_ids=remap_ids
        )

        return converter.convert()
