function r = pad_list_of_sequences(samples, token_size, pre_padding)
    len = [];
    for s = samples
        len(end+1) = length(s);
    end
    max_timesteps = max(len);
    batch_size = length(samples);
    sh = [];
    for sam = samples
        for chk = sam
            s = size(chk{1});
            sh(end+1) = s(1);
        end
    end
    max_event_num = max(sh);
    batch_data = zeros(max_timesteps, batch_size, max_event_num, token_size);
    num_sample = 1;
    chunk_num = 1;
    for action_sample = samples
        num_chunks = length(action_sample);
        for chunk = action_sample
            sz = size(chunk{1});
            chunk_events = sz(1);
            if chunk_events == 0
                continue
            end
            if pre_padding ~= false
                batch_data(end-(num_chunks-chunk_num), num_sample, end+1-chunk_events:end, :) = chunk{1};
            else 
                batch_data(chunk_num, num_sample, 1:chunk_events, :) = chunk{1};
            end
            chunk_num = chunk_num + 1;
        end
        num_sample = num_sample + 1;
    end  
    r = batch_data;
end