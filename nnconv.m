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

    input = [zeros(1,size((input,1)+2*pad);zeros(size(input,1),1),input,zeros(size(input,1),1);zeros(1,size((input,1)+2*pad)];
    output = zeros(size(input,1),size(input,1),1,num_output);
    for i=1:size(input,4)
      for j=1:num_output
        output(:,:,1,j) = conv2(input(:,:,1,i),W(:,:,i,j),'valid') + b(j);
      end
    end

end
