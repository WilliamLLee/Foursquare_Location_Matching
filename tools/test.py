from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('../models/xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("../models/xlm-roberta-base")

# prepare input
text = "Replace me by any text </s><s> you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
# forward pass
output = model(**encoded_input)
