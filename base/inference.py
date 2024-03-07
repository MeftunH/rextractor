import os
import pickle as pickle
import sys

import DataLoader
import torch
import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)

  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])

  return origin_label

def load_test_dataset(dataset_dir, tokenizer, model_type):
  if model_type == 'base':
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))

    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset['id'], tokenized_test, test_label
  elif model_type == 'entity_special':
      test_dataset = load_data(dataset_dir, model_type)
      test_label = list(map(int, test_dataset['label'].values))

      tokenized_test, entity_type = special_tokenized_dataset(test_dataset, tokenizer)
      return test_dataset['id'], tokenized_test, test_label, entity_type
  elif model_type == 'entity_punct':
      test_dataset = load_data(dataset_dir, model_type)
      test_label = list(map(int, test_dataset['label'].values))

      tokenized_test, entity_type = punct_tokenized_dataset(test_dataset, tokenizer)
      return test_dataset['id'], tokenized_test, test_label

def main(CFG):

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  Tokenizer_NAME = CFG['MODEL_NAME']
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  tokenizer = add_token(tokenizer, CFG['MODEL_TYPE'])
  MODEL_NAME = CFG['MODEL_SAVE_DIR']
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  test_dataset_dir = CFG['TEST_PATH']

  if CFG['MODEL_TYPE'] == 'base':
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  elif CFG['MODEL_TYPE'] == 'entity_special':
    model = SepecialEntityBERT(Tokenizer_NAME, model_config, tokenizer) # custom model 에는 내부에 from_pretrained 함수가 없다.
    state_dict = torch.load(f'{MODEL_NAME}/pytorch_model.bin')
    model.load_state_dict(state_dict)

    test_id, test_dataset, test_label, entity_type = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_special_Dataset(test_dataset ,test_label, entity_type)

  elif CFG['MODEL_TYPE'] == 'entity_punct':
    model = SepecialPunctBERT(Tokenizer_NAME, model_config, tokenizer)
    state_dict = torch.load(f'{MODEL_NAME}/pytorch_model.bin')
    model.load_state_dict(state_dict)

    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  model.to(device)

  pred_answer, output_prob = inference(model, Re_test_dataset, device)
  pred_answer = num_to_label(pred_answer)

  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission.csv', index=False)

  print('---- Finish! ----')

  if __name__ == '__main__':
      seed_everything()

      with open('./../module/config.yaml') as f:
          CFG = yaml.safe_load(f)

      main(CFG)