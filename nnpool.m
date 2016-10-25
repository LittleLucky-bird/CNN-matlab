function output = nnpool(input, kernel_size, pad)
    % Your codes here
    % hint:
    %     1. pad zeros on the input's each side
    %     2. use im2col to extract consecutive kernel_size * kernel_size
    %        patches from input
    %     3. get average value in each patch
    %     4. restore the original layout
    output = zeros(size(input,1)/kernel_size, size(input,2)/kernel_size,size(input,3),size(input,4));
    for i=1:size(output,3)
      for j=1:size(output,4)
        val = im2col(input(:,:,i,j),[kernel_size, kernel_size],'distinct');
        val = mean(val);
        output(:,:,i,j) = reshape(val,size(input,1)/kernel_size, size(input,2)/kernel_size);
      end
    end
end
