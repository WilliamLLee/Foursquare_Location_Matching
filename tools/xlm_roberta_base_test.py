from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('../models/xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("../models/xlm-roberta-base")

# prepare input
text = "Replace me by any text </s><s> you'd like."
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
for h in output['hidden_states']:
    print(h.shape)
print(output.keys(), output['hidden_states'][-1].shape)