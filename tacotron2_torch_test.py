import torch
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()
#
# waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
# waveglow = waveglow.remove_weightnorm(waveglow)
# waveglow = waveglow.to('cuda')
# waveglow.eval()
#
# text = "Hello world, I missed you so much."
#
# utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
# sequences, lengths = utils.prepare_input_sequence([text])
#
# with torch.no_grad():
#     mel, _, _ = tacotron2.infer(sequences, lengths)
#     audio = waveglow.infer(mel)
# audio_numpy = audio[0].data.cpu().numpy()
# rate = 22050
#
# from scipy.io.wavfile import write
# write("audio.wav", rate, audio_numpy)
