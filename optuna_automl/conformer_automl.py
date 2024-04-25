import os
import sys
import pickle
import logging

import torch
import optuna
from scipy.stats import gmean
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from distillation_utils import DataProcessing, recognize, evaluate
from conformer_model import ConformerSpeechRecognizer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, concatenate_datasets

os.environ['HF_DATASETS_OFFLINE'] = '1'
automl_logger = logging.getLogger(__name__)


class ConformerAutoML:

    def __init__(self,
                 transformers_cache: str,
                 teacher_model_path: str,
                 list_of_labels_path: str,
                 sberdev_dataset_path: str,
                 test_batch_size: int = 5):

        os.environ['TRANSFORMERS_CACHE'] = transformers_cache
        os.environ['HF_DATASETS_CACHE'] = transformers_cache
        self.test_seeds = [42, 24, 0, 98]
        self.test_batch_size = test_batch_size

        # prepare data
        with open(list_of_labels_path, 'rb') as fp:
            list_of_labels = pickle.load(fp)
        dataset = load_dataset(sberdev_dataset_path)
        datasets_list = list(dataset.values())
        datasets_list = [datasets_list[0], datasets_list[2], datasets_list[1]]
        self.prepared_dataset = concatenate_datasets(datasets_list)
        train_classes = [2, 4, 5, 6, 7, 8]
        self.train_indices = []
        train_labels = []
        test_classes = [0, 1, 3, 9]
        self.test_indices = {}
        for class_label in test_classes:
            self.test_indices[f'{class_label}'] = []
        index_counter = 0
        for i in list_of_labels:
            if i in train_classes:
                self.train_indices.append(index_counter)
                train_labels.append(i)
            if i in test_classes:
                self.test_indices[f'{i}'].append(index_counter)
            index_counter += 1

        # decrease train part
        x_train, x_test, y_train, y_test = train_test_split(self.train_indices,
                                                            train_labels,
                                                            test_size=0.3,
                                                            random_state=42)
        self.train_indices = x_train

        # prepare teacher model
        if not torch.cuda.is_available():
            error_msg = 'CUDA is not available!'
            automl_logger.error(error_msg)
            raise ValueError(error_msg)
        self.cuda_device = torch.device('cuda')
        self.asr_teacher_processor = Wav2Vec2Processor.from_pretrained(teacher_model_path)
        self.asr_teacher_model = Wav2Vec2ForCTC.from_pretrained(teacher_model_path)
        self.asr_teacher_model.to(self.cuda_device)
        self.data_processor = DataProcessing(processor=self.asr_teacher_processor,
                                             teacher=self.asr_teacher_model)
        self.log_interval = 300

    def train_and_evaluate(self,
                           conformer_model_try: ConformerSpeechRecognizer,
                           hyperparams_list: dict):

        current_conformer_model = conformer_model_try
        optimizer = torch.optim.Adam(current_conformer_model.parameters(),
                                     lr=hyperparams_list['learning_rate'])
        loss_fn = torch.nn.KLDivLoss(log_target=True).to(self.cuda_device)
        automl_logger.info('Training is started.')
        automl_logger.info(f'Hyperparameters: {hyperparams_list}')
        evaluation_cers = []

        for test_fold, test_seed in zip(list(self.test_indices.keys()), self.test_seeds):

            test_fold_indices = self.test_indices[test_fold]
            torch.cuda.manual_seed(test_seed)
            torch.backends.cudnn.deterministic = True

            train_loader = DataLoader(
                dataset=self.prepared_dataset,
                collate_fn=self.data_processor.data_collator,
                batch_size=hyperparams_list['minibatch_size'],
                sampler=torch.utils.data.SubsetRandomSampler(self.train_indices),
            )

            test_loader = DataLoader(
                dataset=self.prepared_dataset,
                collate_fn=self.data_processor.data_collator,
                batch_size=self.test_batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(test_fold_indices),
            )

            # training
            current_conformer_model.train()
            step_counter = 0
            all_processed_files_amount = 0
            processed_files_amount = 0
            for batched_data in train_loader:
                if batched_data != 'No data':
                    (spectrograms_padded,
                     spectrogram_masks_padded,
                     emissions_padded,
                     labels_padded,
                     labels_padded_lengths,
                     batch_length) = batched_data
                    spectrograms_padded = spectrograms_padded.to(self.cuda_device)
                    spectrogram_masks_padded = spectrogram_masks_padded.to(self.cuda_device)
                    emissions_padded = emissions_padded.to(self.cuda_device)
                    train_logits = current_conformer_model(spectrograms_padded,
                                                           spectrogram_masks_padded.to(torch.long).sum(dim=-1))
                    masked_targets = emissions_padded[spectrogram_masks_padded]
                    masked_logits = torch.nn.functional.log_softmax(train_logits[spectrogram_masks_padded], dim=-1)
                    loss = loss_fn(masked_logits,
                                   masked_targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step_counter += 1
                    processed_files_amount += batch_length
                    all_processed_files_amount += hyperparams_list['minibatch_size']
                    if step_counter % self.log_interval == 0:
                        bad_files_amount = all_processed_files_amount - processed_files_amount
                        automl_logger.info(
                            f'Step: {step_counter} | KL loss {loss.item()} | Processed files: {processed_files_amount} | Bad files: {bad_files_amount}')

                    del batched_data, spectrograms_padded, spectrogram_masks_padded
                    del emissions_padded, labels_padded, labels_padded_lengths
                    del train_logits, masked_targets, masked_logits

            # evaluation
            automl_logger.info(f'Evaluation started, test fold: {test_fold}')
            current_conformer_model.eval()
            val_asr_results = []
            for val_batched_data in test_loader:
                if val_batched_data != 'No data':
                    (val_spectrograms_padded,
                     val_spectrogram_masks_padded,
                     val_emissions_padded,
                     val_labels_padded,
                     val_labels_padded_lengths,
                     val_batch_length) = val_batched_data
                    val_spectrograms_padded = val_spectrograms_padded.to(self.cuda_device)
                    val_spectrogram_masks_padded = val_spectrogram_masks_padded.to(self.cuda_device)
                    val_labels_padded = val_labels_padded.to(self.cuda_device)
                    val_labels_padded_lengths = val_labels_padded_lengths.to(self.cuda_device)
                    val_batch_res = current_conformer_model(val_spectrograms_padded,
                                                            val_spectrogram_masks_padded.to(torch.long).sum(
                                                                dim=-1))
                    val_asr_results += recognize(
                        outputs=val_batch_res.detach().cpu(),
                        masks=val_spectrogram_masks_padded.cpu(),
                        true_labels=val_labels_padded.cpu(),
                        true_label_lengths=val_labels_padded_lengths.cpu(),
                        processor=self.asr_teacher_processor
                    )
                    del val_spectrograms_padded, val_spectrogram_masks_padded, val_emissions_padded
                    del val_labels_padded, val_batched_data, val_labels_padded_lengths

            # torch.cuda.empty_cache()

            evaluation_cer = evaluate(val_asr_results)
            automl_logger.info(f'Evaluation CER: {evaluation_cer}')
            automl_logger.info('***********************************************')
            evaluation_cers.append(evaluation_cer)

        total_evaluation_cer = gmean(evaluation_cers)
        automl_logger.info(f'Total Character Error Rate:{total_evaluation_cer * 100} %')
        return total_evaluation_cer

    def start(self):

        # optuna tuning
        def objective(trial):
            hidden_layer_size = 2 ** (trial.suggest_int(name='hidden_layer_size', low=6, high=9))
            ffn_dim = 2 ** (trial.suggest_int(name='ffn_dim', low=5, high=10))
            num_heads = 2 ** (trial.suggest_int(name='num_heads', low=1, high=3))
            parameters = {
                'hidden_layer_size': hidden_layer_size,
                'num_heads': num_heads,
                'minibatch_size': trial.suggest_int(name='minibatch_size', low=4, high=16),
                'learning_rate': trial.suggest_float(name='learning_rate', low=1e-6, high=1e-2, log=True),
                'num_layers': trial.suggest_int(name='num_layers', low=4, high=6, log=True),
                'depthwise_conv_kernel_size': trial.suggest_int(name='dc_kernel_size', low=3, high=12, step=2),
                'kernel_size': trial.suggest_int(name='kernel_size', low=3, high=10),
                'ffn_dim': ffn_dim,
                'dropout': trial.suggest_float(name='dropout', low=0.1, high=0.5)
            }
            configured_conformer_model = ConformerSpeechRecognizer(
                kernel_size=parameters['kernel_size'],
                ffn_dim=parameters['ffn_dim'],
                feature_vector_size=13,
                hidden_layer_size=parameters['hidden_layer_size'],
                num_layers=parameters['num_layers'],
                num_heads=parameters['num_heads'],
                dropout=parameters['dropout'],
                depthwise_conv_kernel_size=parameters['depthwise_conv_kernel_size'],
                vocabulary_size=len(self.asr_teacher_processor.tokenizer))
            configured_conformer_model.to(self.cuda_device)
            character_error_rate = self.train_and_evaluate(conformer_model_try=configured_conformer_model,
                                                           hyperparams_list=parameters)
            return character_error_rate

        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(study_name='Conformer_Hyperparameters_AutoML',
                                    direction="minimize",
                                    sampler=sampler)
        study.optimize(objective, n_trials=50)
        automl_logger.info('Best parameters: {})\n'.format(study.best_params))


if __name__ == '__main__':
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler('Conformer_Hyperparameters_AutoML.log')
    file_handler.setFormatter(formatter)
    automl_logger.setLevel(logging.INFO)
    automl_logger.addHandler(stdout_handler)
    automl_logger.addHandler(file_handler)

    conformer_auto_ml = ConformerAutoML(
        transformers_cache='cache',
        teacher_model_path='wav2vec2-large-ru-golos',
        list_of_labels_path='list_of_labels',
        sberdev_dataset_path='sberdevices_golos_10h_crowd',
        test_batch_size=4)
    conformer_auto_ml.start()
