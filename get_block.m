function r = get_block(name, varargin)    
    if  strcmp(name,'MLP')
        a = varargin{1};
        b = varargin{2};
        c = varargin{3};
        d = varargin{4};
        e = varargin{5};
        f = varargin{6};
        g = varargin{7};
        r = MLPBlock(a,b,c,d,e,f,g);
    elseif strcmp(name,'TransformerBlock')
        a = varargin{1};
        b = varargin{2};
        c = varargin{3};
        d = varargin{4};
        e = varargin{5};
        f = varargin{6};
        g = varargin{7};
        h = varargin{8:end};
        r = TransformerBlock(a,b,c,d,e,f,g,h);
    else 
    end
end