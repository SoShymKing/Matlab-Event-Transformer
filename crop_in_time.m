function r = crop_in_time(all_params, total_events)
    validation = all_params.backbone_params.pos_enc_grad;
    augmentation_params = all_params.data_params.augmentation_params;
    num_sparse_frames = augmentation_params.max_sample_len_ms;
    l = size(total_events);
    if l(1) > num_sparse_frames
        if validation == false 
            init = rand .* (length(total_events) - num_sparse_frames);
            last = init + num_sparse_frames;
            total_events = total_events(init:last,:,:,:);
        else 
            init = quorem(sym(length(total_events) - num_sparse_frames), sym(2));
            last = init + num_sparse_frames;
            total_events = total_events(init:last,:,:,:);
        end
    end
    r = total_events;

end