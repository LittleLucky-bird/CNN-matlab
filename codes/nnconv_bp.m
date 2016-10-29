function [down_delta, grad_W, grad_b] = nnconv_bp(input, delta, W, b, pad)
    % Your codes here
    % hint:
    %     follow the formula in slides(page 17).
    %     Generally speaking, the backpropagation process is also
    %     a convolution process, with input and rotated delta.
    %     Gradient checking can assure you of a correct implementation
    input = [zeros(pad,size(input,1)+2*pad,size(input,3),size(input,4));zeros(size(input,1),pad,size(input,3),size(input,4)),input,zeros(size(input,1),pad,size(input,3),size(input,4));zeros(pad,size(input,1)+2*pad,size(input,3),size(input,4))];
    down_delta = zeros(size(input));
    grad_W = zeros(size(W));
    grad_b = zeros(size(b));
    for n=1:size(delta,4)
      for j=1:size(W,4)
        for i=1:size(W,3)
          down_delta(:,:,i,n) = down_delta(:,:,i,n) + conv2(delta(:,:,j,n),W(:,:,i,j),'full');
          grad_W(:,:,i,j) = grad_W(:,:,i,j) + conv2(input(:,:,i,n),rot90(delta(:,:,j,n),2),'valid');
        end
        grad_b(j) = grad_b(j) + sum(sum(delta(:,:,j,n)));
      end
    end

    ddelta = zeros(size(delta));
    for n=1:size(delta,4)
      for i=1:size(W,3)
        ddelta(:,:,i,n) = down_delta(1+pad:end-pad,1+pad:end-pad,i,n);
      end
    end
    down_delta = ddelta;
end
