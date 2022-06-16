#for pre-trained model test
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("../models/xlm-roberta-base")
#tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

# prepare input
text = "Replace me by any text you'd like."
encoded_input = tokenizer(
                text,
                add_special_tokens=True,
                padding = 'max_length',
                truncation = True,
                return_offsets_mapping = False,
                max_length = 150,
                return_token_type_ids = False,
                return_attention_mask = False,
                return_tensors = 'pt',
            )
print(encoded_input)
# forward pass
output = model(**encoded_input,output_hidden_states=True, output_attentions=True)
output2 = model(**encoded_input,output_hidden_states=True, output_attentions=True)
for h in output['hidden_states']:
    print(h.shape)
#print(output.keys(), output['hidden_states'][-1].shape)
print(output.pooler_output.shape)
# print(output2['hidden_states'][-1])

# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')

# # forward pass
# output = model(**encoded_input)
# print(output)
# import torch
# print(encoded_input['input_ids'])
# print(torch.argmax(output['logits'], dim=2))