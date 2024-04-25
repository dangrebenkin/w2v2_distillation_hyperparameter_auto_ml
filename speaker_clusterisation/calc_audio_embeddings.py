import torch
from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector

batch_size = 5
counter = 0
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2-base-superb-sv")
model = Wav2Vec2ForXVector.from_pretrained("wav2vec2-base-superb-sv")


def batch_embeddings(batch_data):
    global counter
    batch_arrays = [i['audio']['array'] for i in batch_data]
    inputs = feature_extractor(
        batch_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    indexed_emds = {}
    for i in range(0, len(batch_arrays)):
        indexed_emds[f'{counter}'] = embeddings[i]
        counter += 1
    return indexed_emds


def generate_embs_dataset(subset):
    global counter
    counter = 0
    dataset = load_dataset('sberdevices_golos_10h_crowd',
                           streaming=True,
                           split=subset)
    data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                shuffle=False,
                collate_fn=batch_embeddings,
                batch_size=batch_size,
                pin_memory=True)
    embeddings_dict = {}
    for completed_batch_data in data_loader:
        embeddings_dict.update(completed_batch_data)
    torch.save(embeddings_dict, f'{subset}.pt')


generate_embs_dataset('train')
generate_embs_dataset('test')
generate_embs_dataset('validation')

