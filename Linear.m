classdef Linear < Layer

    properties
        num_input;
        num_output;

        W;
        b;

        input;
        input_shape;
        output;

        grad_W;
        grad_b;
        diff_W; % last update for W
        diff_b; % last update for b

        delta;
    end

    methods
        function layer = Linear(name, num_input, num_output, init_std)
            layer = layer@Layer(name);
            layer.is_trainable = true;
            layer.num_input = num_input;
            layer.num_output = num_output;

            layer.W = single(random('norm', 0, init_std, num_output, num_input));
            layer.b = zeros(num_output, 1, 'single');
            layer.diff_W = zeros(size(layer.W), 'single');
            layer.diff_b = zeros(size(layer.b), 'single');
        end

        function layer = forward(layer, input)
            layer.input_shape = size(input);
            layer.input = reshape(input,layer.num_input,layer.input_shape(4));
            layer.output = layer.W * layer.input + repmat(layer.b,1,size(layer.input,2));
        end

        function layer = backprop(layer, delta)
            layer.grad_W =   delta * (layer.input)';
            layer.grad_b =   sum(delta,2);
            layer.delta = (layer.W)' * delta;
            layer.delta = reshape(layer.delta ,layer.input_shape);
        end

        function layer = update(layer, config)
            % SGD with momentum and weight decay
            mm = config.momentum;
            lr = config.learning_rate;
            wd = config.weight_decay;

            layer.diff_W = mm * layer.diff_W - lr * (layer.grad_W + wd * layer.W);
            layer.W = layer.W + layer.diff_W;

            layer.diff_b = mm * layer.diff_b - lr * (layer.grad_b + wd * layer.b);
            layer.b = layer.b + layer.diff_b;
        end
    end
end
