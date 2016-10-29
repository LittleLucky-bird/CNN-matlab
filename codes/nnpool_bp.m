function down_delta = nnpool_bp(input, delta, kernel_size, pad)
    % Your codes here
    % hint:
    %     follow the formula in slide (page 19)
    %     Generally speaking, the delta from upper layer is upsampled
    %     averagely to the down_delta in each pooling kernel_size
    down_delta = zeros(size(input));
    for i=1:size(input,3)
      for j=1:size(input,4)
        down_delta(:,:,i,j) = kron(delta(:,:,i,j)/(kernel_size*kernel_size),ones(kernel_size,kernel_size));
      end
    end

end
