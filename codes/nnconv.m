function output = nnconv(input, kernel_size, num_output, W, b, pad)
    % Your codes here
    % hint:
    %     1. first pad zeros on the input's each side
    %     2. convolve input with W
    %           notice the output of j-th filter in W convolved with input
    %           correspond to the j-th channel in output
    %     3. don't forget adding bias
    %
    % ps: there are more than one way in step 2, try to find the fastest method
    output = zeros(size(input,1),size(input,1),num_output,size(input,4));
    input = [zeros(pad,size(input,1)+2*pad,size(input,3),size(input,4));zeros(size(input,1),pad,size(input,3),size(input,4)),input,zeros(size(input,1),pad,size(input,3),size(input,4));zeros(pad,size(input,1)+2*pad,size(input,3),size(input,4))];

    for n=1:size(input,4)
      for j=1:num_output
        for i=1:size(input,3)
          output(:,:,j,n) = conv2(input(:,:,i,n),rot90(W(:,:,i,j),2),'valid') + output(:,:,j,n);
        end
        output(:,:,j,n) = output(:,:,j,n) + b(j);
      end
    end
end
