function [r1, r2, r3]  = collate(batch_samples)
    pols = {};
    pixels = {};
    labels = [];
    num_sample = 0;
    for sample = batch_samples
        s_s = size(sample);
        if isempty(sample) || length(sample(1)) == 0
            continue
        end
        tmp = sample{1};
        pols(end+1) = sample{1};
        tmp = sample{2};
        pixels(end+1) = sample{2};
        labels(end+1) = sample{3};
        num_sample = num_sample + 1;
    end
    if length(pols) == 0
        pols = [];
        pixels = [];
        labels = [];
    end
    ts_s = size(pols{1});
    token_size = ts_s(end);
    pols = pad_list_of_sequences(pols, token_size, true);
    pixels = pad_list_of_sequences(pixels, 2, true);
    r1 = pols;
    r2 = pixels;
    r3 = labels;
end