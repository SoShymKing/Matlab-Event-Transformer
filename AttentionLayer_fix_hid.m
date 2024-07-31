classdef AttentionLayer_fix_hid < nnet.layer.Layer ...
        & nnet.layer.Formattable & nnet.layer.Acceleratable
    % AttentionLayer   Attention layer
    %
    %   To create an attention layer, use attentionLayer.
    %
    %   AttentionLayer properties:
    %       NumHeads               - Number of attention heads
    %       Scale                  - Multiplicative factor for scaling dot
    %                                product of queries and keys
    %       AttentionMask          - Attention mask
    %       DropoutProbability     - Dropout probability for attention
    %                                scores
    %       HasPaddingMaskInput    - Flag indicating layer has padding mask
    %                                input
    %       HasScoresOutput        - Flag indicating layer has attention
    %                                score output
    %       Name                   - Name for the layer
    %       NumInputs              - Number of inputs of the layer
    %       InputNames             - Names of the inputs of the layer
    %       NumOutputs             - Number of outputs of the layer
    %       OutputNames            - Names of the outputs of the layer
    %
    %   Example:
    %       % Create an attention layer with 12 number of heads.
    %
    %       numHeads = 12;
    %       layer = attentionLayer(numHeads);
    %
    %   See also attentionLayer
    
    %   Copyright 2023 The MathWorks, Inc.

    properties(SetAccess = private)
        NumHeads
        Scale
        AttentionMask
    end

    properties
        DropoutProbability
        HasPaddingMaskInput
    end

    properties(SetAccess = private)
        HasScoresOutput
    end

    methods
        function layer = AttentionLayer_fix_hid(name, numHeads, scale, attentionMask, ...
                dropoutProbability, hasPaddingMaskInput, hasScoresOutput)
            layer.Name = name;
            layer.NumHeads = numHeads;
            layer.Scale = scale;
            layer.AttentionMask = attentionMask;
            layer.DropoutProbability = dropoutProbability;
            layer.HasPaddingMaskInput = hasPaddingMaskInput;
            layer.HasScoresOutput = hasScoresOutput;
            layer.Description = iGetMessageString('nnet_cnn:layer:AttentionLayer:OneLineDisplay', ...
                int2str(layer.NumHeads));
            layer.Type = iGetMessageString('nnet_cnn:layer:AttentionLayer:Type');
            layer = layer.setInputOutputNames;
        end

        function varargout = forward(layer, varargin)
            dropoutProb = layer.DropoutProbability;
            varargout = layer.doForward(dropoutProb, varargin{:});
        end

        function varargout = predict(layer, varargin)
            dropoutProb = 0;
            varargout = layer.doForward(dropoutProb, varargin{:});
        end

        function layer = initialize(layer, varargin)
            % predict can throw error messages with long stack trace
            % containing internal dlarray methods used in
            % dlarray.attention. Use try/catch to clean up the error
            % message
            try
                args = cellfun(@iLayoutToDlarray, varargin, 'Uniform', 0);
                dropoutProb = 0;
                layer.doForward(dropoutProb, args{:});
            catch exception
                % Reduce stack so that the first function call is to
                % dlarray/attention
                stack = exception.stack;
                ind = iFindAttentionFunInStack(stack);
                err = struct("identifier", exception.identifier, "message", exception.message, "stack", stack(ind:end));
                error(err);
            end
        end

        function layer = set.DropoutProbability(layer, value)
            validateattributes(value, {'numeric'}, {'scalar','real','finite','nonsparse','nonnegative','<', 1});
            value = nnet.internal.cnn.layer.util.convertToDouble(value);
            layer.DropoutProbability = value;
        end
    end

    methods(Access=private)
        function layer = setInputOutputNames(layer)
            inputNames = {'query', 'key', 'value'};
            outputNames = {'out'};
            if layer.HasPaddingMaskInput
                inputNames = [inputNames, 'mask'];
            end
            if layer.HasScoresOutput
                outputNames = [outputNames, 'scores'];
            end
            layer.InputNames = inputNames;
            layer.OutputNames = outputNames;
        end

        function out = doForward(layer, dropoutProb, varargin)
            Q = varargin{1};
            K = varargin{2};
            V = varargin{3};
            nvArgs = {'Scale', layer.Scale, 'AttentionMask', layer.AttentionMask, 'DropoutProbability', dropoutProb};
            if layer.HasPaddingMaskInput
                paddingMask = varargin{4};
                nvArgs = [nvArgs, {'PaddingMask', paddingMask}];
            end
            if layer.HasScoresOutput
                [out{1}, scores] = attention(matlab.lang.internal.move(Q), ...
                    matlab.lang.internal.move(K), matlab.lang.internal.move(V), layer.NumHeads, nvArgs{:});
                out{2} = dlarray(scores, 'UUUU');
            else
                out{1} = attention(matlab.lang.internal.move(Q), K, V, layer.NumHeads, nvArgs{:});
            end
        end
    end

    methods(Static, Hidden)
        function n = matlabCodegenRedirect(~)
            n = 'nnet.internal.cnn.coder.layer.AttentionLayer';
        end
    end
end

function messageString = iGetMessageString(varargin)
messageString = getString(message(varargin{:}));
end

function out = iLayoutToDlarray(layout)
% Convert networkDataLayout object to dlarray
sz = layout.Size;
sz(isnan(sz)) = 1;
out = dlarray(ones(sz), layout.Format);
end

function ind = iFindAttentionFunInStack(stack)
% Find index corresponding to dlarray/attention call in stack
ind = 1;
for i=1:numel(stack)
    if strcmp(stack(i).name,'attention')
        ind = i;
        return
    end
end
end