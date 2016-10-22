classdef Relu < Layer
    properties
        input;
        output;
        delta;
    end

    methods
        function layer = Relu(name)
            layer = layer@Layer(name);
        end

        function layer = forward(layer, input)
            % Your codes here
            layer.input = input;
            layer.output = single(max(0,input));
        end

        function layer = backprop(layer, delta)
            % Your codes here
            A = zeros(size(layer.input));
            A(layer.output > 0) = 1;
            layer.delta = delta .* A;
        end
    end
end
