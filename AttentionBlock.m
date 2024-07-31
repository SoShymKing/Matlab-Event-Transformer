classdef AttentionBlock < nnet.layer.Layer & nnet.layer.Acceleratable

    properties
        net = dlnetwork;
        mask = "none";
        in_mask = "none";
        attL;
        dim;
        num_heads
        mask_ptr = libpointer
        q_mask_ptr = libpointer
    end

    properties (Learnable)
    end
    methods
        function layer = AttentionBlock(Name, opt_dim, heads, dropout, att_dropout, varargin)
            layer.Name = Name;
            layer.Description = "AttentionBlock";
            layer.NumInputs = 2;
            layer.dim = opt_dim;
            layer.num_heads = heads;

            tempNet = functionLayer(@(X1,X2)layer.mask_check(X1,X2), NumInputs=2,NumOutputs=2, Name="func", InputNames=["inp_kv","inp_q"],OutputNames = ["inp_kv","inp_q"], Formattable = 1, Acceleratable=1); % inputLayer([NaN NaN opt_dim],"TBC","Name","z");
            layer.net = addLayers(layer.net,tempNet);
            tempNet = [
                layerNormalizationLayer("Name","layernorm")];
            layer.net = addLayers(layer.net,tempNet);
            tempNet = layerNormalizationLayer("Name","layernorm_1");
            layer.net = addLayers(layer.net,tempNet);
            layer.attL = attentionLayer_fix(heads,"Name","attention",'AttentionMask',layer.mask);
            tempNet = [
                layer.attL
                dropoutLayer(att_dropout,"Name","dropout_2")];
            layer.net = addLayers(layer.net,tempNet);
            tempNet = [
                additionLayer(2,"Name","addition")
                layerNormalizationLayer("Name","layernorm_2")
                dropoutLayer(dropout,"Name","dropout")
                functionLayer(@(X)layer.reshape_data(X), NumInputs=1,NumOutputs=1,Name="rs_1", Formattable = 1, Acceleratable=1)
                fullyConnectedLayer(opt_dim,"Name","afc")
                geluLayer("Name","gelu")
                layerNormalizationLayer("Name","layernorm_3")
                functionLayer(@(X)layer.reshape_data(X), NumInputs=1,NumOutputs=1,Name="rs_2", Formattable = 1, Acceleratable=1)
                fullyConnectedLayer(opt_dim,"Name","afc_1")
                geluLayer("Name","gelu_1")
                dropoutLayer(dropout,"Name","dropout_1")
                functionLayer(@(X)layer.reshape_data(X), NumInputs=1,NumOutputs=1,Name="rs_3", Formattable = 1, Acceleratable=1)
                fullyConnectedLayer(opt_dim,"Name","afc_2")];
            layer.net = addLayers(layer.net,tempNet);
            tempNet = additionLayer(2,"Name","addition_1");
            layer.net = addLayers(layer.net,tempNet);

            layer.net = connectLayers(layer.net,"layernorm","attention/key");
            layer.net = connectLayers(layer.net,"layernorm","attention/value");
            layer.net = connectLayers(layer.net,"func/inp_kv","layernorm");
            layer.net = connectLayers(layer.net,"func/inp_q","layernorm_1");
            layer.net = connectLayers(layer.net,"func/inp_q","addition/in1");
            layer.net = connectLayers(layer.net,"layernorm_1","attention/query");
            layer.net = connectLayers(layer.net,"dropout_2","addition/in2");
            layer.net = connectLayers(layer.net,"addition","addition_1/in2");
            layer.net = connectLayers(layer.net,"afc_2","addition_1/in1");

            clear tempNet;

            layer.net = networkLayer(layer.net,Name = layer.Name);
        end
        function Y = reshape_data(layer,X)
            size(X);
            Y = dlarray(X,"CBT");
        end
        function [Y1,Y2] = mask_check(layer,inp_kv, inp_q)
            if isNull(layer.mask_ptr) == false || isempty(layer.mask_ptr.DataType) == false
                mask_size = size(layer.mask_ptr.Value);
                bsz = laeyr.dim;
                src_len = layer.dim;
                if mask_size == [bsz, src_len]
                    tmp = reshape(layer.mask_ptr.Value,bsz, 1, 1, src_len);
                    tmp_ar = zeros(bsz, layer.num_heads, 1, src_len);
                    for i = 1:layer.num_heads
                        tmp_ar(:,i,:,:) = tmp;
                    end
                    mask_t = reshape(tmp_ar,bsz * layer.num_heads, 1, src_len);
                    if isNull(layer.q_mask_ptr) == true || isempty(layer.q_mask_ptr.Value)
                        layer.in_mask = mask_t;
                    elseif class(layer.q_mask_ptr.Value) == "logical"
                        layer.in_mask = mask_t | layer.q_mask_ptr.Value;
                    else
                        for j=1:mask_size(1)
                            for k = 1:mask_size(2)
                                if mask_t(j,k) > 0
                                    q_mask(j,k) = single(-Inf);
                                end
                            end
                        end
                        layer.in_mask = q_mask;
                    end
                    layer.mask = layer.in_mask;
                else 
                    layer.mask = "none";
                end
            else 
                layer.mask = "none";
            end
            layer.attL.HasPaddingMaskInput = layer.mask;
            kv_s = size(inp_kv);
            q_s = size(inp_q);
            if kv_s(1)~=q_s(1) || kv_s(2)~=q_s(2)
                if isempty(find(kv_s==q_s(1),1)) || isempty(find(kv_s==q_s(2),1))
                else
                    inp_kv = reshape(inp_kv, kv_s(kv_s==q_s(finddim(inp_q,"C"))), kv_s(kv_s==q_s(finddim(inp_q,"B"))),[]);
                end
            end
            Y1=dlarray(inp_kv,"CBT");
            Y2=dlarray(inp_q,"CBT");
        end
        function Y = predict(layer,X1, X2)
            Y = predict(layer.net,X1,X2);
        end
    end
end