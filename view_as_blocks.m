function arr_out = view_as_blocks(arr_in, block_shape)
    if length(block_shape) ~= length(size(arr_in))
    end
    arr_shape = size(arr_in);
    if sum(rem(arr_shape,block_shape)) ~= 0
    end
    new_shape = horzcat(quorem(sym(arr_shape),sym(block_shape)),block_shape);
    arr_out = reshape(arr_in,new_shape);
end