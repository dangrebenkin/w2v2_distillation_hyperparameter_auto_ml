import gc
from typing import Dict, List

import jiwer
import jiwer.transforms as tr
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def recognize(outputs: torch.Tensor,
              masks: torch.BoolTensor,
              true_labels: torch.LongTensor,
              true_label_lengths: torch.LongTensor,
              processor: Wav2Vec2Processor) -> List[Dict[str, str]]:
    assert len(outputs.shape) == 3
    assert len(masks.shape) == 2
    assert outputs.shape[0] == masks.shape[0]
    assert outputs.shape[1] == masks.shape[1]
    assert len(true_labels.shape) == 2
    assert len(true_label_lengths.shape) == 1
    assert true_labels.shape[0] == outputs.shape[0]
    assert true_label_lengths.shape[0] == outputs.shape[0]
    output_lengths = torch.sum(masks, dim=-1).to(torch.long)
    res = []
    for idx in range(outputs.shape[0]):
        true_ids = true_labels[idx, 0:true_label_lengths[idx]].numpy()
        predicted_ids = torch.argmax(outputs[idx, 0:output_lengths[idx]], dim=-1).numpy()
        true_text = processor.decode(true_ids, group_tokens=False)
        predicted_text = processor.decode(predicted_ids)
        res.append({'true': true_text, 'predicted': predicted_text})
    return res


def levenshtein(seq1: List[str],
                seq2: List[str]) -> float:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return float(matrix[size_x - 1, size_y - 1])


def evaluate(results: List[Dict[str, str]]) -> float:
    n_total_char_dist = 0.0
    n_total_chars = 0.0
    for cur_pair in results:
        true_text = list(' '.join(cur_pair['true'].lower().strip().split()))
        predicted_text = list(' '.join(cur_pair['predicted'].lower().strip().split()))
        cur_dist = levenshtein(predicted_text, true_text)
        cur_char_number = len(true_text)
        n_total_char_dist += cur_dist
        n_total_chars += cur_char_number
    cer = n_total_char_dist / float(n_total_chars)
    return cer


class ReplaceYo(tr.AbstractTransform):
    def process_string(self, s: str):
        return s.replace('ё', 'е')

    def process_list(self, inp: List[str]):
        outp = []
        for sentence in inp:
            outp.append(sentence.replace('ё', 'е'))
        return outp


normalize_transcription = jiwer.Compose([
    tr.ToLowerCase(),
    tr.RemovePunctuation(),
    tr.RemoveWhiteSpace(replace_by_space=True),
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
    ReplaceYo()
])


class DataProcessing:
    def __init__(self,
                 processor: Wav2Vec2Processor,
                 teacher: Wav2Vec2ForCTC,
                 n_mfcc: int = 13,
                 target_sr: int = 16000):
        self.target_sr = target_sr
        self.processor = processor
        self.teacher = teacher
        self.mfcc_calculator_ = torchaudio.transforms.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_mels': 64,
                'n_fft': 512,
                'win_length': 400,
                'hop_length': 320,
                'center': False,
                'power': 2,
                'f_min': 200,
                'f_max': 5000
            }).to(self.teacher.device)

    def data_collator(self,
                      batch_data: list):

        sounds = []
        annotations = []
        for i in batch_data:
            try:
                sound = np.array(i['audio']['array']).astype(np.float32)
                annotation = normalize_transcription(i['transcription'])
                sounds.append(sound)
                annotations.append(annotation)
            except BaseException:
                continue

        if len(sounds) > 0:

            batch_length = len(sounds)

            # calc emissions
            processed = self.processor(
                sounds, sampling_rate=self.target_sr,
                padding='longest',
                return_tensors="pt"
            )
            input_values = processed.input_values.float().to(self.teacher.device)
            attention_mask = processed.attention_mask.to(self.teacher.device)
            with torch.no_grad():
                feat_extract_output_lengths = self.teacher._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
                emissions = torch.log_softmax(self.teacher(input_values,
                                                           attention_mask=attention_mask).logits,
                                              dim=-1).cpu()
            del input_values, attention_mask

            # calc mfcc
            max_sound_length = max(map(lambda it: it.shape[0], sounds))
            sound_batch = np.zeros((batch_length, max_sound_length), dtype=np.float32)
            for sample_idx, sound in enumerate(sounds):
                n = sound.shape[0]
                sound_batch[sample_idx, 0:n] = sound.astype(np.float32)
            with torch.no_grad():
                features = self.mfcc_calculator_(torch.from_numpy(sound_batch).to(self.teacher.device))
            features = np.moveaxis(features.cpu().numpy(), 1, 2)
            prepared_inputs = []
            for sample_idx in range(0, batch_length):
                n = feat_extract_output_lengths[sample_idx]
                new_mfcc = features[sample_idx]
                if new_mfcc.shape[0] < n:
                    prepared_inputs.append(
                        np.vstack(
                            (
                                new_mfcc,
                                np.zeros((n - new_mfcc.shape[0], new_mfcc.shape[1]), dtype=new_mfcc.dtype)
                            )
                        )
                    )
                elif new_mfcc.shape[0] == n:
                    prepared_inputs.append(new_mfcc)
                else:
                    prepared_inputs.append(new_mfcc[0:n])
                del new_mfcc
            del features, sound_batch, max_sound_length

            # calc input ids
            prepared_labels = []
            with self.processor.as_target_processor():
                tokenized_annotations = self.processor(
                    annotations,
                    padding='do_not_pad',
                    return_tensors='np'
                )
            for sample_idx in range(0, batch_length):
                prepared_labels.append(tokenized_annotations['input_ids'][sample_idx])

            # padding
            spectrograms = [torch.from_numpy(i).to(torch.float) for i in prepared_inputs]
            spectrogram_masks = [torch.ones(size=(x.size(0),), dtype=torch.bool) for x in spectrograms]
            prepared_labels = [torch.from_numpy(np.array(i)).to(torch.long) for i in prepared_labels]
            emissions = [emission.to(torch.float) for emission in emissions]

            spectrograms_pad = torch.nn.utils.rnn.pad_sequence(
                spectrograms,
                batch_first=True, padding_value=0
            )
            spectrogram_masks_pad = torch.nn.utils.rnn.pad_sequence(
                spectrogram_masks,
                batch_first=True, padding_value=False
            )
            emissions_pad = torch.nn.utils.rnn.pad_sequence(
                emissions,
                batch_first=True, padding_value=0.0
            )
            labels_pad = torch.nn.utils.rnn.pad_sequence(
                prepared_labels,
                batch_first=True, padding_value=-100
            )
            labels_lengths = torch.LongTensor([y.size(0) for y in prepared_labels])
            del spectrograms, spectrogram_masks, emissions, prepared_labels
            gc.collect()
            return spectrograms_pad, spectrogram_masks_pad, emissions_pad, labels_pad, labels_lengths, batch_length
        else:
            return 'No data'
