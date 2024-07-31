function [net,info] = trainnet_t(varargin)
%

%   Copyright 2022-2023 The MathWorks, Inc.

narginchk(4,5);
% Guess which kind of syntax the user was intending based on which input
% matches the layers input.
if iIsLayersArgument(varargin{2})  % try to match {data, layers, loss, options}
    narginchk(4,4);
    data = varargin{1};
elseif iIsLayersArgument(varargin{3}) % try to match {predictors, responses, layers, loss, options}
    narginchk(5,5);
    data = varargin(1:2);
else
    error(message('deep:train:InvalidNetwork'));
end

[net, loss, options] = deal(varargin{end-2:end});

try
    net = deep.internal.train.validateLayersAndConstructNetwork(net);
    loss = deep.internal.train.validateAndStandardizeLoss(loss);
    options = deep.internal.train.validateOptions(options);
    
    [mbq,networkInfo,metadata] = deep.internal.train.createMiniBatchQueue(net,data,...
        MiniBatchSize = options.MiniBatchSize,...
        SequenceLength = options.SequenceLength, ...
        SequencePaddingValue = options.SequencePaddingValue, ...
        SequencePaddingDirection = options.SequencePaddingDirection,...
        InputDataFormats = options.InputDataFormats,...
        TargetDataFormats = options.TargetDataFormats,...
        PartialMiniBatch="discard");

    % Error if underlying data is gpuArray but training options specify CPU
    % training
    if metadata.IsGpuArray && ismember(options.ExecutionEnvironment, ["cpu", "parallel-cpu"])
        error(message('deep:train:GpuArrayDataWithCPUExecutionEnvironment'))
    end

    [net,loss] = deep.internal.train.prepareForTraining(mbq,net,loss,options);

    [net,info] = deep.internal.train.trainnet(mbq, net, loss, options, ...
        TrainingDataMetaData = metadata,...
        NetworkInfo = networkInfo);
catch err
    nnet.internal.cnn.util.rethrowDLExceptions(err);
end
end

function tf = iIsLayersArgument(layers)
tf = isa(layers, 'nnet.cnn.layer.Layer') || isa(layers, 'dlnetwork');
end
