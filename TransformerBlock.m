classdef TransformerBlock < nnet.layer.Layer & nnet.layer.Acceleratable
    properties
        net = dlnetwork;
        mask_ptr = libpointer
        q_mask_ptr = libpointer
    end
    methods
        function layer = TransformerBlock(Name, opt_dim, latent_blocks, dropout, att_dropout, heads, cross_heads, varargin)
            layer.Name = Name;
            layer.Description = "TransformerBlock";
            layer.NumInputs = 2;
            tmpnet = AttentionBlock("Att_0", opt_dim, cross_heads, dropout, att_dropout);
            tmpnet.q_mask_ptr = layer.q_mask_ptr;
            tmpnet.mask_ptr = layer.mask_ptr;
            layer.net = addLayers(layer.net, tmpnet.net);
            for tmp = 1:latent_blocks
                tmpl = AttentionBlock("Att_"+tmp, opt_dim, heads, dropout, att_dropout);
                tmpl.q_mask_ptr = layer.q_mask_ptr;
                tmpl.mask_ptr = libpointer;
                layer.net = addLayers(layer.net, tmpl.net);
                layer.net = connectLayers(layer.net,"Att_"+(tmp-1),strcat("Att_"+tmp,'/func/inp_kv'));
                layer.net = connectLayers(layer.net,"Att_"+(tmp-1),strcat("Att_"+tmp,'/func/inp_q'));
            end
            clear tmpnet;
            layer.net = networkLayer(layer.net,Name = layer.Name);

        end
        
        function Y = predict(layer,X1,X2, varargin)
            Y = predict(layer.net,X1,X2, OutputDataFormats = 'TBC');
        end

    end
end