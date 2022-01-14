import torch
from fairseq.models.wav2vec import Wav2VecModel

cp = torch.load()
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()
