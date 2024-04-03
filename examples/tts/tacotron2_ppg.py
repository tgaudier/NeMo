# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
import torch

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import Tacotron2PPGModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


# hydra_runner is a thin NeMo wrapper around Hydra
# It looks for a config named tacotron2.yaml inside the conf folder
# Hydra parses the yaml and returns it as a Omegaconf DictConfig
@hydra_runner(config_path="conf", config_name="tacotron2_ppg")
def main(cfg):
    model = Tacotron2PPGModel(cfg=cfg.model)
    model.load_state_dict(torch.load("/lium/home/tgaudier/checkpoints/tacotron2.ckpt")["state_dict"], strict=False)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
