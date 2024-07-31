% https://github.com/louislva/deepmind-perceiver/blob/eb668c818eceab957713091ad1e244b14f039f7e/perceiver/positional_encoding.py#L6
function r = fourier_features(shape, bands)
    dims = length(shape);
    lins = {};
    for  n = 1:length(shape)
        lins(end+1)={linspace(-1.0,1.0,shape(n))};
    end
    [X,Y]=meshgrid(lins{:});
    pos = cat(1,shiftdim(X,-1),shiftdim(Y,-1));
    pos1 = shiftdim(pos,-1);
    pos2 = repmat(pos1,cat(2,[bands],ones(1,length(size(pos)))));
    logs = logspace(log10(1.0),log10(shape(1)/2),bands)/log(exp(1));
    logs = reshape(logs,[bands 1]);
    pos_size = size(pos2);
    pos_size(1) = 1;
    band_frequencies = repmat(logs,pos_size);
    rst = reshape(band_frequencies .* pi .* pos2,cat(2,[dims*bands],shape));
    rst = cat(1,sin(rst),cos(rst));
    r = rst;
end
    
