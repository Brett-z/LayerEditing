
def layer_removal(
    model,
    prune_list,
):
    """
    A Model Agnostic Implementation Method of directly remove specified layers from the LLM
    """

    layers = model.model.layers
    sorted_list = sorted(prune_list, reverse=True) 
    prune_layer_num = len(sorted_list)
    first_del_layer = min(sorted_list)
    
    # remove layers from back to front
    for prune_layer in sorted_list:
        del(layers[prune_layer])

    for i in range(first_del_layer, model.config.num_hidden_layers - prune_layer_num):
        model.model.layers[i].self_attn.layer_idx = model.model.layers[i].self_attn.layer_idx - prune_layer_num
        # print(model.model.layers[i].self_attn.layer_idx)