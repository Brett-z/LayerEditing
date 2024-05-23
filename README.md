# LayerEditing
We have implemented a Model Agnostic function called  `layer_removal` , which is designed to directly remove specified layers from the LLM.  This function is suitable for the transformers library from HuggingFace. At the same time, we also provide a method for recovering removed layers called `layer_restoration` .

To remove a layer, we need to consider its importance. We drew inspiration from the theories presented in the paper below:

- "[ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/pdf/2403.03853)"

## Use Example

```python
import layer_editing
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Define a list to store the layer numbers that need to be pruned
prune_list = [26,27]

# Call the layer_removal function and pass the model and the list as parameters to the function
restore_list = layer_editing.layer_removal(model, prune_list)

# Load the model into cuda after running the function
model = model.cuda()

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# Restore
layer_editing.layer_restoration(model, prune_list, restore_list)
```

