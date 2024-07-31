classdef LatentEmbsCompressor < nnet.layer.Layer & nnet.layer.Acceleratable

    properties
        net = dlnetwork;
        clf
    end

    methods
        function layer = LatentEmbsCompressor(Name, ltv_n, opt_dim, clf_mode, embs_norm, varargin)
            layer.Description = "AttentionBlock";
            layer.Name = Name;
            layer.clf = clf_mode;
            layer.net = addLayers(layer.net,functionLayer(@(X)layer.check_in(X), NumInputs=1,NumOutputs=1,Name="chk_1", Formattable=1, Acceleratable=1));
            if embs_norm
                layer.net = addLayers(layer.net,layerNormalizationLayer("Name","layernorm"));
            end
            tempNet = [
                fullyConnectedLayer(opt_dim,"Name","lfc")
                reluLayer("Name","relu")];
            layer.net = addLayers(layer.net,tempNet);
            layer.net = addLayers(layer.net,functionLayer(@(X1)layer.mean_output(X1), NumInputs=1,NumOutputs=1, Name="check_mean", InputNames="x", OutputNames ="y"));
            if embs_norm
                layer.net = connectLayers(layer.net,"chk_1","layernorm");
                layer.net = connectLayers(layer.net,"layernorm","lfc");
            else
                layer.net = connectLayers(layer.net,"chk_1","lfc");
            end
            layer.net = connectLayers(layer.net,"relu","check_mean");
            clear tempNet;
            layer.net = networkLayer(layer.net,Name = layer.Name);
        end
        function Y = check_in(layer,X)
            Y = dlarray(X,"TBC");
        end
        function Y = mean_output(layer,X)
            if strcmp(layer.clf,'gap')
                Y = mean(X);
            else
                Y = X;
            end
        end
        function Y = predict(layer,X)
            Y = predict(layer.net,X);
        end
    end
end