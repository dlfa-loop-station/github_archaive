from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

# 모델 초기화
BASE_DIR = "gs://download.magenta.tensorflow.org/models/music_vae/colab2"

drums_models = {}
# One-hot encoded.
drums_config = configs.CONFIG_MAP['cat-drums_2bar_small']
drums_models['drums_2bar_oh_lokl'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/checkpoints/drums_2bar_small.lokl.ckpt')
drums_models['drums_2bar_oh_hikl'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/checkpoints/drums_2bar_small.hikl.ckpt')

# Multi-label NADE.
drums_nade_reduced_config = configs.CONFIG_MAP['nade-drums_2bar_reduced']
drums_models['drums_2bar_nade_reduced'] = TrainedModel(drums_nade_reduced_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/checkpoints/drums_2bar_nade.reduced.ckpt')
drums_nade_full_config = configs.CONFIG_MAP['nade-drums_2bar_full']
drums_models['drums_2bar_nade_full'] = TrainedModel(drums_nade_full_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/checkpoints/drums_2bar_nade.full.ckpt')
