# LayerRemoval
We have implemented a Model Agnostic function called  `layer_removal` , which is designed to directly remove specified layers from the LLM.  This function is suitable for the transformers library from HuggingFace.

## Use Example

```python
import layer_removal
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Define a list to store the layer numbers that need to be pruned
prune_list = [26,27]

# Call the layer_removal function and pass the model and the list as parameters to the function
layer_removal(model, prune_list)

# Load the model into cuda after running the function
model = model.cuda()

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

