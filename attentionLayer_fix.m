function layer = attentionLayer_fix(numHeads, nameValueArgs)
% attentionLayer   Attention layer
%   
%   layer = attentionLayer(numHeads) creates an attention layer that
%   computes dot product attention between the inputs: queries, keys and
%   values. numHeads specifies the number of attention heads.
%
%   layer = attentionLayer(numHeads, Name=Value)
%   specifies additional options using one or more name-value arguments:
%
%     Scale               - Multiplicative factor for scaling the dot
%                           product of queries and keys, specified as one
%                           of these values:
%                           "auto" - Multiply the dot product by 1/sqrt(D),
%                                    where D is size of the number of
%                                    channels of keys divided by numHeads.
%                                    This is the value proposed in [1].
%                           Scalar - Multiply the dot product by the
%                                    specified scalar value. Value 1
%                                    implies no scaling.
%                           The default value is "auto".
%
%     HasPaddingMaskInput - Flag indicating whether the layer has an extra
%                           input that represents the padding mask,
%                           specified as 1 (true) or 0 (false). If
%                           HasPaddingMaskInput is 1 (true), then the layer
%                           has an extra input with the name "mask". The
%                           padding mask input must be a logical array or
%                           numeric array consisting of 0 and 1. The
%                           software prevents and allows attention to
%                           elements in keys-values pairs when the
%                           corresponding element in the padding mask is 0
%                           and 1 respectively.
%                           The default value is 0 (false).
%
%     HasScoresOutput     - Flag indicating whether the layer has an
%                           extra output that represents attention scores,
%                           specified as 1 (true) or 0 (false). If
%                           HasScoresOutput is 1 (true), then the layer has
%                           an output with the name "scores".
%                           The default value is 0 (false).
%
%     AttentionMask       - Mask preventing attention to elements in
%                           certain positions in input data key-value
%                           pairs, specified as one of the following:
%                           "none"   - Do not prevent attention to elements
%                                      with respect to their positions. If
%                                      HasPaddingMaskInput is 1, the
%                                      function prevents attention to
%                                      padding elements only.
%                           "causal" - Prevent an element in a position M
%                                      in the spatial or time dimension of
%                                      queries from providing attention to
%                                      an element in position N > M in the
%                                      corresponding dimension of input
%                                      key-value pairs. Use this option for
%                                      auto-regressive models.
%                           Array    - Logical or numeric array consisting
%                                      of 0 and 1. Prevent attention to
%                                      elements of input data key-value
%                                      pairs in positions corresponding to
%                                      the positions of 0 in the array. The
%                                      array must be a Nk-by-Nq matrix or a
%                                      Nk-by-Nq-by-numObservations array,
%                                      where Nk is the size of the spatial
%                                      or time dimension in keys, Nq is the
%                                      size of the corresponding dimension
%                                      in queries, and  numObservations is
%                                      the size of the observation
%                                      dimension in the input data.
%                           The default value is "none".
%
%     DropoutProbability  - Dropout probability for attention weights,
%                           specified as a nonnegative scalar less than 1.
%                           The default value is 0.
%
%     Name                - Name for the layer.
%                           The default value is "".
%
%   Example:
%       % Create an attention layer with 12 number of heads.
%
%       numHeads = 12;
%       layer = attentionLayer(numHeads);
%
%   1. Vaswani et al. Attention is all you need. Advances in Neural
%      Information Processing Systems (2017).
%
%   See also nnet.cnn.layer.AttentionLayer, selfAttentionLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2023 The MathWorks, Inc.

arguments
    numHeads (1,1) {mustBeNumeric, mustBeInteger, mustBeNonempty, mustBePositive, mustBeNonsparse}
    nameValueArgs.Scale                  = 'auto'
    nameValueArgs.HasPaddingMaskInput    {iAssertBinary} = false
    nameValueArgs.HasScoresOutput        {iAssertBinary} = false
    nameValueArgs.AttentionMask          = 'none'
    nameValueArgs.DropoutProbability     {iAssertValidDropoutProbability} = 0
    nameValueArgs.Name                   {iAssertValidLayerName} = ''
end

% Convert arguments to canonical form
args = nameValueArgs;
args.NumHeads = numHeads;
args = iConvertToCanonicalForm(args);

% Construct the layer
layer = AttentionLayer_fix_hid(args.Name, args.NumHeads, args.Scale, ...
    args.AttentionMask, args.DropoutProbability, args.HasPaddingMaskInput, args.HasScoresOutput);
end

function iAssertBinary(value)
nnet.internal.cnn.options.OptionsValidator.assertBinary(value);
end

function iAssertValidDropoutProbability(value)
validateattributes(value, {'numeric'}, {'scalar','real','finite','nonsparse','nonnegative','<', 1});
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

function scale = iValidateScale(scale)
try
    if ischar(scale) || isstring(scale)
        scale = validatestring(scale, {'auto'});
    else
        validateattributes(scale, 'numeric', {'scalar','real','finite','nonsparse'});
        scale = iConvertToDouble(scale);
    end
catch err
    throwAsCaller(MException(message('nnet_cnn:layer:AttentionLayer:InvalidScale')));
end
end

function mask = iValidateAttentionMask(mask)
try
    if ischar(mask) || isstring(mask)
        mask = validatestring(mask, {'none', 'causal'});
    else
        validateattributes(mask, {'numeric', 'logical'}, {'nonempty', 'nonsparse', 'binary'});
        mask = iConvertToDouble(mask);
    end
catch err
    throwAsCaller(MException(message('nnet_cnn:layer:AttentionLayer:InvalidAttentionMask')));
end
end

function inputArguments = iConvertToCanonicalForm(params)
% Make sure integer values are converted to double and strings to char
% vectors
inputArguments = struct;
inputArguments.NumHeads = iConvertToDouble(params.NumHeads);
inputArguments.Scale = iValidateScale(params.Scale);
inputArguments.AttentionMask = iValidateAttentionMask(params.AttentionMask);
inputArguments.DropoutProbability = iConvertToDouble(params.DropoutProbability);
inputArguments.HasPaddingMaskInput = logical(params.HasPaddingMaskInput);
inputArguments.HasScoresOutput = logical(params.HasScoresOutput);
inputArguments.Name = char(params.Name);
end

function value = iConvertToDouble(value)
value = nnet.internal.cnn.layer.util.convertToDouble(value);
end
