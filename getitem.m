function [r1, r2, r3] = getitem(all_params, total_events, total_pdls, return_sparse_array, varargin)
    if return_sparse_array == true
        return_sparse_array = true;
    else
        return_sparse_array = false;
    end
    events = rotator(total_events);
    chunk_len_ms = all_params.data_params.chunk_len_ms;
    chunk_len_us = chunk_len_ms*1000;
    sparse_frame_len_us = 1000;
    sparse_frame_len_ms = quorem(sym(sparse_frame_len_us),sym(1000));
    chunk_size = quorem(sym(chunk_len_us),sym(sparse_frame_len_us));
    preproc_polarity = all_params.data_params.preproc_polarity;
    num_extra_chunks = all_params.data_params.num_extra_chunks;
    bins = all_params.data_params.bins;
    patch_size = all_params.data_params.patch_size;
    if contains(preproc_polarity,'1')
        original_event_size = 1;
    else 
        original_event_size = 2;
    end
    preproc_event_size = original_event_size*bins;
    if all_params.data_params.min_activations_per_patch > 0 && all_params.data_params.min_activations_per_patch <= 1 
        min_activations_per_patch = floor(all_params.data_params.min_activations_per_patch .* patch_size .* patch_size+1);
    else
        min_activations_per_patch = 0;
    end
    min_patches_per_chunk = all_params.data_params.min_patches_per_chunk;
    augmentation_params = all_params.data_params.augmentation_params;
    validation = all_params.backbone_params.pos_enc_grad;
    if isfield(augmentation_params,'h_flip') == true
        h_flip = augmentation_params.h_flip;
    else
        h_flip = false;
    end
    acc = total_pdls(1);
    brk = total_pdls(2);
    if isfield(augmentation_params,'max_sample_len_ms') == true  && augmentation_params.max_sample_len_ms ~= -1
        events = crop_in_time(all_params, events);
    end
    if isfield(augmentation_params,'random_frame_size') == true  && ~isnan(augmentation_params.random_frame_size)
        events = crop_in_space(all_params, events);
    end
    if validation == false && h_flip==true && rand>0.5
        events = flip(events,3);
    end
    total_pixels = {};
    total_polarity = {};
    if sum(acc,"all")/length(acc)<0.05 && sum(brk,"all")/length(brk)<0.05
        label = 1;
    elseif sum(acc,"all")/length(acc) > sum(brk,"all")/length(brk)
        label = 0;
    elseif sum(acc,"all")/length(acc) < sum(brk,"all")/length(brk)
        label = 2;   
    else
        label = 1;
    end
    current_chunk = [];
    s = size(events);
    sf_num = s(1)-1;
    while sf_num >= 0
        if isempty(current_chunk) == true 
            current_chunk = events(max([1, sf_num - chunk_size + 1]):sf_num,:,:,:);
            current_chunk = flip(current_chunk, 1);
            sf_num = sf_num - chunk_size;
            if contains(preproc_polarity,'1')
                tmp_s = size(current_chunk);
                current_chunk = sum(current_chunk, length(tmp_s));
            end
        else
            sf = events(max([1, sf_num - chunk_size + 1]):sf_num,:,:,:);
            sf = flip(sf, 1);
            sf_num = sf_num- num_extra_chunks;
            if contains(preproc_polarity,'1')
                tmp_s = size(sf);
                sf = sum(sf, length(tmp_s));
            end
            current_chunk = cat(1,current_chunk, sf);

        end
        tmp_s = size(current_chunk);        
        if tmp_s(1) < bins
            continue
        end
        bins_init = tmp_s(1);
        bins_step = quorem(sym(bins_init),sym(bins));
        chunk_candidate = {};
        steps = 0:bins_step:bins_init-1;

        ib_num = 1;
        for i = steps(1:bins)
            if ib_num == bins  
                chunk_candidate{end+1}= squeeze(sum(current_chunk(i + 1:end, : , : , :),1));
            else
                step = bins_step;
                chunk_candidate{end+1} = squeeze(sum(current_chunk(i + 1:i + step, : , : , :),1));
            end
            ib_num = ib_num + 1;
        end

        chunk_candidate = cat(length(tmp_s),chunk_candidate{1},chunk_candidate{2});
        chunk_size = size(chunk_candidate);
        chunk_candidate = reshape(chunk_candidate,chunk_size(1),chunk_size(2), chunk_size(3)*chunk_size(4));
        block_shape = [patch_size, patch_size, preproc_event_size];
        arr_shape = size(chunk_candidate);
        if sum(rem(arr_shape, block_shape)) ~= 0  
            print(block_shape,arr_shape)
        end
        polarity = view_as_blocks(chunk_candidate, block_shape);
        shp = size(polarity);
        inds = sum(polarity,length(shp)) ~= 0;
        inds = reshape(inds,[shp(1),shp(2),patch_size*patch_size]);
        itmp = size(inds);
        inds = reshape(sum(inds,length(itmp)),[shp(1)*shp(2),1,[]]);
        inds = squeeze(inds);
        inds = inds >= min_activations_per_patch;
        if sum(inds) == 0
        end
        if min_patches_per_chunk && sum(inds) < min_patches_per_chunk
        end
        polarity = reshape(polarity,shp(1)*shp(2),patch_size*patch_size*preproc_event_size); % self.token_dim
        pixels = [];
        for i = 0:patch_size:arr_shape(1)-1
            for j = 0:patch_size:arr_shape(2)-1
                pixels(end+1,:) = [quorem(sym(i+patch_size),sym(2)),quorem(sym(j+patch_size),sym(2))];
            end
        end
        inds = find(inds~=0);
        inds = inds(:,1);
        if validation == false ...
            && length(inds)>0 ...
            && isfield(augmentation_params,'drop_token') == true ...
            && augmentation_params.drop_token{1} ~= 0
            inds = randsample(inds,max(1, int(len(inds)*(1-augmentation_params.drop_token{1}))), false);
        end
        px_s = size(pixels);
        pl_s = size(polarity);
        tmp_px = [];
        tmp_pl = [];
        for i = 1:px_s(2)
            tmp = pixels(:,i);
            tmp_px(:,i) = tmp(inds);
        end
        for i = 1:pl_s(2)
            tmp = polarity(:,i);
            tmp_pl(:,i) = tmp(inds);
        end

        if isfield(preproc_polarity,'log') == true
            tmp_pl = np.log(tmp_pl + 1);
        else
        end
        total_polarity{end+1} = [tmp_pl];
        total_pixels{end+1} = [tmp_px];
        current_chunk = [];
    end
    if isfield(augmentation_params,'random_shift') && augmentation_params.random_shift == true
        evt_s = size(events);
        total_pixels = shift(all_params, total_pixels, evt_s(2:end));
    end
    r1 = total_polarity;
    r2 = total_pixels;
    r3 = label;
end