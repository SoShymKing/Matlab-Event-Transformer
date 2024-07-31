function r = crop_in_space(all_params, total_events)
    validation = all_params.backbone_params.pos_enc_grad;
    patch_size = all_params.data_params.patch_size;
    height = all_params.backbone_params.pos_encoding.params.shape(2);
    width = all_params.backbone_params.pos_encoding.params.shape(1);
    augmentation_params = all_params.data_params.augmentation_params;
    x_lims = [int32(width*augmentation_params.random_frame_size), width];
    y_lims = [int32(height*augmentation_params.random_frame_size), height];
    s = size(total_events);
    y_size = s(2);
    x_size = s(3);
    if validation == false
        new_x_size = x_lims(1) + (x_lims(2)-x_lims(1)) * rand;
        new_y_size = y_lims(1) + (y_lims(2)-y_lims(1)) * rand;
        if patch_size ~= 1
            new_x_size = new_x_size - rem(new_x_size, patch_size);
            new_y_size = new_y_size - rem(new_y_size, patch_size);
        end
        x_init = rand * (x_size - new_x_size+1);
        x_end = x_init + new_x_size;
        y_init = rand * (y_size - new_y_size+1);
        y_end = y_init + new_y_size;
        total_event = total_events(:, y_init:y_end, x_init:x_end, :);
    else 
        new_x_size = quorem(sym(x_lims(1) + x_lims(2)),sym(2));
        new_y_size = quorem(sym(y_lims(1) + y_lims(2)),sym(2));
        if patch_size ~= 1
            new_x_size = new_x_size - rem(new_x_size, patch_size);
            new_y_size = new_y_size - rem(new_y_size, patch_size);
        end
        x_init = quorem(sym(x_size - new_x_size),sym(2));
        x_end = x_init + new_x_size;
        y_init = quorem(sym(y_size - new_y_size),sym(2));
        y_end = y_init + new_y_size;
        total_event = total_events(:, y_init+1:y_end, x_init+1:x_end, :);
    end
    r = total_event;
end