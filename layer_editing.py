
def layer_removal(
    self,
    prune_list,
):
    """
    A Model Agnostic Implementation Method of directly removing specified layers from the LLM
    """
    layers = self.model.model.layers
    sorted_list = sorted(prune_list, reverse=True) 
    prune_layer_num = len(sorted_list)
    for_restore = []
    
    # remove layers from back to front
    for prune_layer in sorted_list:
        for_restore.append(layers[prune_layer])
        del(layers[prune_layer])

    sorted_list = sorted(prune_list) 

    for j in range(0, prune_layer_num):
        start = sorted_list[j] - j
        for i in range(start, self.model.config.num_hidden_layers - prune_layer_num):
            self.model.model.layers[i].self_attn.layer_idx = self.model.model.layers[i].self_attn.layer_idx - 1

    return for_restore

def layer_restoration(
    self, 
    prune_list, 
    for_restore,
):
    """
    An Implementation Method of restoring deleted layers
    """
    insert_times = len(prune_list) - 1
    sorted_list = sorted(prune_list)
    prune_idx = 0

    while insert_times >= 0:
        self.model.model.layers.insert(sorted_list[prune_idx], for_restore[insert_times])
        insert_times = insert_times - 1
        prune_idx = prune_idx + 1

    for i in range(0, len(self.model.model.layers)):
        self.model.model.layers[i].self_attn.layer_idx = i
