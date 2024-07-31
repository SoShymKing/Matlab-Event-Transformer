function r = shift(all_params, total_pixels, cropped_shape)
    height = all_params.backbone_params.pos_encoding.params.shape(2);
    width = all_params.backbone_params.pos_encoding.params.shape(1);
    validation = all_params.backbone_params.pos_enc_grad;
    patch_size = all_params.data_params.patch_size;
    height_diff = height - cropped_shape(1);
    width_diff = width - cropped_shape(2);
    if validation == false
        if height_diff ~= 0.0
            new_height_init = rand * height_diff;
        else 
            new_height_init = 0;
        end
      if width_diff ~= 0.0
        new_width_init = rand*width_diff;
      else 
          new_width_init = 0;
      end
    else
        new_height_init = quorem(sym(height_diff),sym(2));
        new_width_init = quorem(sym(width_diff),sym(2));
    end
    new_height_init = new_height_init - rem(new_height_init, patch_size); 
    new_width_init = new_width_init - rem(new_width_init, patch_size);
    for i = 1:length(total_pixels)
        tmp = total_pixels{i};
        tmp(:,1) = tmp(:,1) + new_height_init; 
        tmp(:,2) = tmp(:,2) + new_width_init; 
        total_pixels{i} = tmp;
    end
    r =  total_pixels;
end