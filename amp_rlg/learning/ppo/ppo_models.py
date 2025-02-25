from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ModelPPOContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return
    
    # called by train/play codes to construct network. self.model = self.network.build(net_config)
    def build(self, config):
        net = self.network_builder.build('ppo', **config) # call ppo_network_builder's build function.
        for name, _ in net.named_parameters():
            print(name)

        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)

        return self.Network(net, obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)


    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            return
        
        def forward(self, input_dict):
            result = super().forward(input_dict)
            return result