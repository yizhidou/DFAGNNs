import dataclasses

import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Any, Iterator, List, Union
from clrs._src import dataset
from clrs._src import yzd_settings, yzd_utils
from clrs._src import yzd_samplers


@dataclasses.dataclass
class YZDDatasetConfig(tfds.core.BuilderConfig):
    #     这我不太知道...这能放sample_id吗？
    name: str = ''  # task_name
    split: str = ''
    sample_id_savepath: str = ''
    seed: int = 0
    max_iteration: int = 0
    sourcegraph_dir: str = ''
    dataset_savedir: Union[None, str] = None
    if_sync: bool = False
    if_idx_reorganized: bool = True
    if_save: bool = False


# def initiate_builder_configs(train_id_list,
#                              val_id_list,
#                              test_id_list,
#                              seed: int,
#                              source_graph_dir,
#                              dataset_savedir: Union[None, str] = None,
#                              if_sync: bool = False,
#                              if_idx_reorganized: bool = True,
#                              if_save: bool = False
#                              ):
#     default_builder_configs = []
#     for task_name in ['yzd_liveness', 'yzd_dominance', 'yzd_reachability']:
#         default_builder_configs.append(YZDDatasetConfig(name=task_name,
#                                                         split='train',
#                                                         sample_id_list=train_id_list,
#                                                         seed=seed,
#                                                         sourcegraph_dir=source_graph_dir,
#                                                         dataset_savedir=dataset_savedir,
#                                                         if_sync=if_sync,
#                                                         if_idx_reorganized=if_idx_reorganized,
#                                                         if_save=if_save
#                                                         ))
#         default_builder_configs.append(YZDDatasetConfig(name=task_name,
#                                                         split='val',
#                                                         sample_id_list=val_id_list,
#                                                         seed=seed,
#                                                         sourcegraph_dir=source_graph_dir,
#                                                         dataset_savedir=dataset_savedir,
#                                                         if_sync=if_sync,
#                                                         if_idx_reorganized=if_idx_reorganized,
#                                                         if_save=if_save
#                                                         ))
#         default_builder_configs.append(YZDDatasetConfig(name=task_name,
#                                                         split='test',
#                                                         sample_id_list=test_id_list,
#                                                         seed=seed,
#                                                         sourcegraph_dir=source_graph_dir,
#                                                         dataset_savedir=dataset_savedir,
#                                                         if_sync=if_sync,
#                                                         if_idx_reorganized=if_idx_reorganized,
#                                                         if_save=if_save
#                                                         ))
#     return default_builder_configs


def unpack_feedback(feed_back: yzd_samplers.Feedback):
    unpacked_data = dict()
    unpacked_data.update({'input_' + t.name: t.data for t in feed_back.features.inputs})
    unpacked_data['lengths'] = feed_back.features.lengths
    unpacked_data.update({'output_' + t.name: t.data for t in feed_back.outputs})
    unpacked_data.update({
        'hint_' + t.name: t.data for t in feed_back.features.hints})
    return unpacked_data


class YZDDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    BUILDER_CONFIGS = []

    def _info(self) -> tfds.core.DatasetInfo:
        if tf.io.gfile.exists(self.data_dir):
            info = tfds.core.DatasetInfo(builder=self)
            info.read_from_directory(self.data_dir)
            return info
        sampler, _ = self._dataset_sampler()
        sampled_data = sampler.next(batch_size=1)
        unpacked_fb = unpack_feedback(sampled_data)
        data = {k: dataset._correct_axis_filtering(v, 0, k)
                for k, v in unpacked_fb}
        data_info = {
            k: tfds.features.Tensor(shape=v.shape, dtype=tf.dtypes.as_dtype(
                v.dtype)) for k, v in data.items()}
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(data_info),
        )

    def get_sample_ids_split(self, sample_id_savepath) -> Iterator[str]:
        with open(sample_id_savepath) as file_reader:
            for line in file_reader.readlines():
                yield line.strip()

    def _split_generators(
            self,
            dl_manager: tfds.download.DownloadManager,
    ):
        return {self._builder_config.split: self._generate_examples()}

    def _dataset_sampler(self):
        sample_id_list = []
        with open(self.builder_config.sample_id_savepath) as sample_id_reader:
            for line in sample_id_reader.readlines():
                sample_id_list.append(line.strip())
        sampler, _ = yzd_samplers.build_yzd_sampler(task_name=self.builder_config.name,
                                                    sample_id_list=sample_id_list,
                                                    seed=self.builder_config.seed,
                                                    sample_loader=yzd_utils.SampleLoader(
                                                        sample_path_processor=yzd_utils.SamplePathProcessor(
                                                            sourcegraph_dir=self.builder_config.sourcegraph_dir,
                                                            dataset_savedir=self.builder_config.dataset_savedir),
                                                        max_iteration=self.builder_config.max_iteration))
        return sampler, len(sample_id_list)

    def _generate_examples(self):
        task_name = self.builder_config.name
        split_name = self.builder_config.split
        # num_samples = len(self.builder_config.sample_id_list)
        sampler, num_samples = self._dataset_sampler()
        for sample_idx in range(num_samples):
            # 其实我这里的和CLRSDataset产生的样本类型是不一样的
            # 我的是Feedback，它的看上去是把Feedback给unpack成一个dict了
            sampled_data = sampler.next(batch_size=1)
            unpacked_fb = unpack_feedback(sampled_data)
            data = {k: dataset._correct_axis_filtering(v, 0, k)
                    for k, v in unpacked_fb}
            yield str(sample_idx), data


if __name__ == '__main__':
    # sample_id_list = ['poj104_103.12003.4', 'poj104_103.12005.2', 'poj104_103.12009.0', 'poj104_103.12014.7',
    #                   'poj104_103.12016.4']
    sample_id_savepath = '/Users/yizhidou/Documents/ProGraMLTestPlayground/test_sample_id.txt'
    test_config = YZDDatasetConfig(name='yzd_liveness',
                                   split='train',
                                   sample_id_savepath=sample_id_savepath,
                                   seed=0,
                                   max_iteration=5,
                                   sourcegraph_dir='/Users/yizhidou/Documents/ProGraMLTestPlayground/TestOutputFiles/poj104_103/programl_downloaded/',
                                   dataset_savedir=None,
                                   if_sync=False,
                                   if_idx_reorganized=True,
                                   if_save=False)
    test_dataset = YZDDataset(config=test_config)
