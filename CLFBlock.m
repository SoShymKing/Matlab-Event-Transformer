classdef CLFBlock < nnet.layer.Layer & nnet.layer.Acceleratable
        % & nnet.layer.Formattable ... % (Optional) 
        % & nnet.layer.Acceleratable % (Optional)

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
        net = dlnetwork;
    end

    methods
        function layer = CLFBlock(Name, ipt_dim, opt_classes, varargin)
            layer.Description = "CLFBlock";
            layer.Name = Name;
            tempNet = [
                functionLayer(@(X) layer.pre_process(X),Name = "pre", Formattable = 1, Acceleratable=1)
                fullyConnectedLayer(ipt_dim,"Name","fc")
                reluLayer("Name","relu")
                fullyConnectedLayer(ipt_dim,"Name","fc_1")
                reluLayer("Name","relu_1")
                fullyConnectedLayer(opt_classes,"Name","fc_2")
                softmaxLayer("Name","softmax")];
            layer.net = addLayers(layer.net,tempNet);
            % 헬퍼 변수 정리
            clear tempNet;
            layer.net = networkLayer(layer.net,Name=layer.Name);
        end
        function Y = pre_process(layer,X)
            Y = dlarray(X,"TBC");
        end
        function Y = predict(layer,X)
            Y = predict(layer.net,X);
        end
    end
end