classdef SoftmaxLoss < handle
    properties
        name;
    end

    methods
        function layer = SoftmaxLoss(name)
            layer = layer@handle();
        end

        function loss = forward(layer, input, target)
            % Your codes here
            % hint:
            %     1. calculate probability from input using Softmax form.
            %        Notice: how to avoid overflow in exponential?
            %     2. loss = sum(target * -log(probability)), where target
            %        is one-hot encoding form label
            probability = exp(input);
            probability_sum = sum(probability);
            probability  = probability ./ repmat(probability_sum,10,1);
            loss = - sum(sum( log( probability(target == 1 ) ) ));
        end

        function delta = backprop(layer, input, target)
            % Your codes here
            probability = exp(input);
            probability_sum = sum(probability);
            probability  = probability ./ repmat(probability_sum,10,1);
            delta = probability - target ;
        end
    end
end
